"""FastAPI serving application for the VideoAnomalyDetector model.

The model is loaded from the MLflow model registry at startup and kept in
memory for low-latency inference.  Clients POST MP4 video files to
/analyze and receive per-segment anomaly scores plus an aggregate verdict.

Environment variables
---------------------
MODEL_NAME          Registered MLflow model name
MODEL_ALIAS         MLflow model alias (default: champion)
MODEL_TAG           Backward-compatible alias env var name
MLFLOW_TRACKING_URI MLflow tracking server URL
NUM_SEGMENTS        Number of temporal segments used during inference
FRAMES_PER_SEGMENT  Frames sampled per segment
ANOMALY_THRESHOLD   Score above which a clip is flagged as anomalous
SAVE_LOCALLY        Set to "true" to write results to the outputs/ directory
OUTPUT_DIR          Directory used when SAVE_LOCALLY is enabled (default: outputs)
"""

from __future__ import annotations

import asyncio
import base64
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime
from io import BytesIO
from pathlib import Path

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image, ImageDraw
from transformers import VideoMAEImageProcessor


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL_NAME = os.environ.get("MODEL_NAME", "industrial-padim-anomaly-detector-R03")
MODEL_ALIAS = os.environ.get("MODEL_ALIAS") or os.environ.get("MODEL_TAG", "champion")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
SAVE_LOCALLY = os.environ.get("SAVE_LOCALLY", "false").lower() == "true"
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "outputs"))

# ---------------------------------------------------------------------------
# Global state (initialised once at startup)
# ---------------------------------------------------------------------------

_model: torch.nn.Module | None = None
_preproc: dict | None = None  # keys: mean, std, resize_to
_device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_model_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _preproc

    _reload_model_from_registry(MODEL_NAME, MODEL_ALIAS)
    yield

    print("Shutting down serving app.")


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="OmniSight Anomaly Detector",
    description=(
        "Serve a VideoMAE-based Multiple Instance Learning anomaly detection model. "
        "POST an MP4 clip to /analyze to receive per-segment anomaly scores."
    ),
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_model_hyperparams() -> None:
    """Populate NUM_SEGMENTS, FRAMES_PER_SEGMENT and ANOMALY_THRESHOLD from the
    MLflow model registry, falling back to environment variables when the
    values are not found.

    The training script stores these under two possible locations:
    * ``ModelInfo.metadata``  — when ``mlflow.pytorch.log_model`` is called
      with ``metadata={"num_segments": …, "frames_per_segment": …,
      "test_best_threshold": …}``
    * Run params / tags      — when ``mlflow.log_params`` / ``mlflow.set_tags``
      are used inside the training run.
    """
    global NUM_SEGMENTS, FRAMES_PER_SEGMENT, ANOMALY_THRESHOLD

    client = mlflow.MlflowClient()
    model_version = client.get_model_version_by_alias(MODEL_NAME, MODEL_ALIAS)
    model_source = model_version.source

    # ── 1. Try model metadata (cheapest lookup) ──────────────────────────────
    try:
        info = mlflow.models.get_model_info(model_source)
        meta = info.metadata or {}
        if "num_segments" in meta:
            NUM_SEGMENTS = int(meta["num_segments"])
        if "frames_per_segment" in meta:
            FRAMES_PER_SEGMENT = int(meta["frames_per_segment"])
        if "test_best_threshold" in meta:
            ANOMALY_THRESHOLD = float(meta["test_best_threshold"])
        print(f"Hyperparams from model metadata: {meta}")
    except Exception as exc:
        print(f"Could not read model metadata ({exc}); trying run params …")
        meta = {}

    # ── 2. Fill any still-missing values from the producing run's params ─────
    missing = (
        "num_segments" not in meta
        or "frames_per_segment" not in meta
        or "test_best_threshold" not in meta
    )
    if missing:
        try:
            run = client.get_run(model_version.run_id)
            if run:
                params = run.data.params
                if "num_segments" not in meta and "num_segments" in params:
                    NUM_SEGMENTS = int(params["num_segments"])
                if "frames_per_segment" not in meta and "frames_per_segment" in params:
                    FRAMES_PER_SEGMENT = int(params["frames_per_segment"])
                if (
                    "test_best_threshold" not in meta
                    and "test_best_threshold" in params
                ):
                    ANOMALY_THRESHOLD = float(params["test_best_threshold"])
                print(f"Hyperparams from run {model_version.run_id} params: {params}")
        except Exception as exc:
            print(f"Could not read run params ({exc}); using env-var defaults.")

    print(
        f"Effective hyperparams — NUM_SEGMENTS={NUM_SEGMENTS}, "
        f"FRAMES_PER_SEGMENT={FRAMES_PER_SEGMENT}, "
        f"ANOMALY_THRESHOLD={ANOMALY_THRESHOLD}"
    )


def _reload_model_from_registry(model_name: str, model_alias: str) -> None:
    """Load model + preprocessing config from the MLflow registry."""
    global _model, _preproc, MODEL_NAME, MODEL_ALIAS, _device

    from anomaly_detector.model import VideoAnomalyDetector  # noqa: F401

    # VideoAnomalyDetector must be importable so MLflow can deserialise the
    # cloudpickle artifact. The import above registers the class in sys.modules
    # before torch.load / cloudpickle.loads reconstructs the model object.
    MODEL_NAME = model_name
    MODEL_ALIAS = model_alias
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_uri = f"models:/{MODEL_NAME}@{MODEL_ALIAS}"

    print(f"Downloading model from {model_uri}...")
    local_path = mlflow.artifacts.download_artifacts(artifact_uri=model_uri)

    # 3. Load from the local downloaded path
    model = mlflow.pytorch.load_model(model_uri=local_path, map_location=_device)
    model.to(_device)
    model.eval()

    # Resolve inference hyperparameters from registry metadata/run params.
    _load_model_hyperparams()

    # Build preprocessing config that matches the training pipeline.
    proc = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    mean = torch.tensor(proc.image_mean).view(3, 1, 1)
    std = torch.tensor(proc.image_std).view(3, 1, 1)
    if "shortest_edge" in proc.size:
        h = w = proc.size["shortest_edge"]
    else:
        h, w = proc.size["height"], proc.size["width"]
    preproc = {"mean": mean, "std": std, "resize_to": (h, w)}

    # Atomic swap so request handlers always see a fully initialised state.
    _model = model
    _preproc = preproc
    print(
        f"Model ready on {_device}. resize={h}x{w}, "
        f"segments={NUM_SEGMENTS}x{FRAMES_PER_SEGMENT} frames, "
        f"threshold={ANOMALY_THRESHOLD}."
    )


def _preprocess_video(path: str) -> torch.Tensor:
    """Decode an MP4 file and return a model-ready tensor of shape (1, N, T, C, H, W)."""
    frames, _, _ = torchvision.io.read_video(path, output_format="TCHW", pts_unit="sec")
    # frames: (T, C, H, W) uint8

    total_frames = frames.shape[0]
    if total_frames == 0:
        raise ValueError("Video contains no decodable frames.")

    mean = _preproc["mean"]
    std = _preproc["std"]
    resize_to = _preproc["resize_to"]

    # Sample NUM_SEGMENTS evenly-spaced start positions across the video.
    start_indices = torch.linspace(0, max(total_frames - 1, 0), NUM_SEGMENTS).long()

    segments: list[torch.Tensor] = []
    for idx in start_indices:
        s = idx.item()
        e = min(s + FRAMES_PER_SEGMENT, total_frames)
        clip = frames[s:e]  # (T', C, H, W) uint8

        # Resize all frames at once (T' is treated as the batch dim by interpolate).
        clip = F.interpolate(
            clip.float(), size=resize_to, mode="bilinear", align_corners=False
        ).byte()

        # Pad the tail clip if shorter than FRAMES_PER_SEGMENT frames.
        if clip.shape[0] < FRAMES_PER_SEGMENT:
            if clip.shape[0] > 0:
                pad = clip[-1:].expand(FRAMES_PER_SEGMENT - clip.shape[0], -1, -1, -1)
                clip = torch.cat([clip, pad], dim=0)
            else:
                clip = torch.zeros(
                    (FRAMES_PER_SEGMENT, 3, *resize_to), dtype=torch.uint8
                )

        # Normalise: [0, 255] → [0, 1] → VideoMAE mean/std
        clip = clip.float() / 255.0
        clip = (clip - mean) / std  # (T, C, H, W)
        segments.append(clip)

    # (N, T, C, H, W) → add batch dim → (1, N, T, C, H, W)
    return torch.stack(segments).unsqueeze(0)


# ---------------------------------------------------------------------------
# Grad-CAM helpers
# ---------------------------------------------------------------------------

# Cached 256-entry jet colormap built on first use (no matplotlib dependency).
_JET_CMAP: np.ndarray | None = None


def _jet_colormap(x: np.ndarray) -> np.ndarray:
    """Apply a jet-like colormap to a 2-D float array in [0, 1].

    Returns an (H, W, 3) uint8 array.
    """
    global _JET_CMAP
    if _JET_CMAP is None:
        t = np.linspace(0.0, 1.0, 256)
        r = np.clip(1.5 - np.abs(4.0 * t - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(4.0 * t - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(4.0 * t - 1.0), 0.0, 1.0)
        _JET_CMAP = (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)
    return _JET_CMAP[(x * 255).astype(np.uint8)]


def _gradcam_for_segment(clips: torch.Tensor) -> torch.Tensor:
    """Compute a Grad-CAM heatmap for a single pre-processed video segment.

    VideoMAE represents video as a grid of spatiotemporal patch tubelets and
    produces one token per tube.  We hook the last encoder layer, capture the
    token activations and their gradients w.r.t. the segment's anomaly score,
    then apply the standard Grad-CAM weighting over the hidden dimension.
    The resulting (t_p, h_p, w_p) map is upsampled via trilinear interpolation
    to the original (T, H, W) frame resolution.

    Parameters
    ----------
    clips:
        Pre-processed segment tensor of shape ``(1, T, C, H, W)``.

    Returns
    -------
    torch.Tensor
        Heatmap of shape ``(T, H, W)`` with values normalised to ``[0, 1]``.
    """
    cfg = _model.backbone.config
    patch_size: int = cfg.patch_size  # typically 16
    tubelet_size: int = cfg.tubelet_size  # typically 2

    T, H, W = clips.shape[1], clips.shape[-2], clips.shape[-1]
    t_p = T // tubelet_size
    h_p = H // patch_size
    w_p = W // patch_size

    # Detach from any previous graph and enable grad tracking so the backbone
    # forward pass is recorded in the autograd DAG.  Backbone *parameters*
    # remain frozen (requires_grad=False); we only need gradient flow through
    # the activations up to the classifier.
    clips_for_grad = clips.detach().requires_grad_(True)

    activation_store: list[torch.Tensor | None] = [None]

    def _fwd_hook(
        module: torch.nn.Module, inp: tuple, out: tuple | torch.Tensor
    ) -> None:
        act = out[0] if isinstance(out, tuple) else out
        act.retain_grad()  # preserve gradient for this non-leaf tensor
        activation_store[0] = act

    handle = _model.backbone.encoder.layer[-1].register_forward_hook(_fwd_hook)
    try:
        with torch.enable_grad():
            backbone_out = _model.backbone(pixel_values=clips_for_grad)
            feat = backbone_out.last_hidden_state.mean(dim=1)  # (1, D)
            score = _model.classifier(feat).squeeze()  # scalar
            score.backward()
    finally:
        handle.remove()

    act = activation_store[0]
    if act is None or act.grad is None:
        # Fallback: return a uniform map if something went wrong
        return torch.ones(T, H, W)

    # ----- Grad-CAM formula for patch tokens --------------------------------
    # Treat hidden dimension D as "channels" and patch index as spatial.
    # alpha_d  = mean over all patches of  ∂score/∂A_{p,d}
    # cam_p    = ReLU( Σ_d  alpha_d · A_{p,d} )
    grad = act.grad  # (1, num_patches, D)
    alpha = grad.mean(dim=1, keepdim=True)  # (1, 1, D)
    cam = F.relu((alpha * act).sum(dim=-1))  # (1, num_patches)
    cam = cam.squeeze(0)  # (num_patches,)

    # Normalise to [0, 1]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    expected = t_p * h_p * w_p
    if cam.shape[0] != expected:
        # Unexpected token count (e.g. CLS token present) — fall back gracefully
        cam_3d = torch.ones(t_p, h_p, w_p)
    else:
        cam_3d = cam.view(t_p, h_p, w_p)

    # Upsample from patch grid to full frame resolution
    cam_up = F.interpolate(
        cam_3d.unsqueeze(0).unsqueeze(0).float(),
        size=(T, H, W),
        mode="trilinear",
        align_corners=False,
    ).squeeze()  # (T, H, W)

    return cam_up.detach().cpu()


def _overlay_heatmap(
    raw_frames: torch.Tensor,
    cam: torch.Tensor,
    alpha: float = 0.45,
) -> list[str]:
    """Blend a Grad-CAM heatmap over raw video frames.

    Parameters
    ----------
    raw_frames:
        Uint8 tensor of shape ``(T, C, H, W)`` — original decoded frames.
    cam:
        Float tensor of shape ``(T, H, W)`` — normalised heatmap in ``[0, 1]``.
    alpha:
        Blending weight for the colourised heatmap overlay.

    Returns
    -------
    list[str]
        One base64-encoded JPEG string per frame.
    """
    frames_b64: list[str] = []
    for i in range(raw_frames.shape[0]):
        frame_np = (
            raw_frames[i].permute(1, 2, 0).numpy().astype(np.float32)
        )  # (H, W, 3)
        heat_np = cam[i].numpy()  # (H, W)
        heat_rgb = _jet_colormap(heat_np).astype(np.float32)  # (H, W, 3)
        blended = (
            ((1.0 - alpha) * frame_np + alpha * heat_rgb).clip(0, 255).astype(np.uint8)
        )
        buf = BytesIO()
        Image.fromarray(blended).save(buf, format="JPEG", quality=85)
        frames_b64.append(base64.b64encode(buf.getvalue()).decode())
    return frames_b64


# ---------------------------------------------------------------------------
# Local output helpers
# ---------------------------------------------------------------------------


def _create_result_video(
    run_dir: Path,
    result: dict,
    raw_frames_list: list[torch.Tensor],
    cam_list: list[torch.Tensor],
    fps: int = 8,
) -> None:
    """Write a side-by-side MP4 to *run_dir/result.mp4*.

    Each frame is composed of:
    * **Left half** — original decoded video frame.
    * **Right half** — jet-coloured Grad-CAM heatmap (black for normal segments).
    * **Title banner** — shows the segment anomaly score and verdict.
    """
    banner_h = 36
    all_frames: list[np.ndarray] = []

    for seg, seg_raw, cam in zip(result["segments"], raw_frames_list, cam_list):
        seg_score = seg["segment_score"]
        is_anom = seg["is_anomalous"]
        title = (
            f"Anomaly score: {seg_score:.4f}  =>  "
            f"{'Anomalous' if is_anom else 'Normal'}"
        )

        for t in range(seg_raw.shape[0]):
            left = seg_raw[t].permute(1, 2, 0).numpy()  # (H, W, 3) uint8
            heat = _jet_colormap(cam[t].numpy())  # (H, W, 3) uint8
            side_by_side = np.concatenate([left, heat], axis=1)  # (H, 2W, 3)

            # Prepend title banner
            banner = np.zeros((banner_h, side_by_side.shape[1], 3), dtype=np.uint8)
            frame = np.concatenate([banner, side_by_side], axis=0)

            img = Image.fromarray(frame)
            ImageDraw.Draw(img).text((8, 8), title, fill=(255, 255, 255))
            all_frames.append(np.array(img))

    if all_frames:
        video_tensor = torch.from_numpy(np.stack(all_frames))  # (T, H, W, C)
        out_path = run_dir / "result.mp4"
        torchvision.io.write_video(str(out_path), video_tensor, fps=fps)
        print(f"Result video saved to {out_path}")


def _save_results_locally(
    filename: str,
    result: dict,
) -> Path:
    """Persist inference results to OUTPUT_DIR for easy local inspection.

    Layout::

        outputs/
          {timestamp}_{stem}/
            summary.json          ← scores, flags, threshold
            segment_{i:02d}/
              frame_{j:03d}.jpg   ← JPEG frames (heatmap overlay for anomalous
                                     segments, plain frame for normal ones)

    Returns the path to the run directory that was created.
    """
    stem = Path(filename).stem if filename else "upload"
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_DIR / f"{ts}_{stem}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Summary JSON (everything except the bulky base64 frame data)
    summary = {k: v for k, v in result.items() if k != "segments"}
    summary["segments"] = [
        {k: v for k, v in seg.items() if k != "frames_base64_jpeg"}
        for seg in result["segments"]
    ]
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-segment frames
    for seg in result["segments"]:
        seg_dir = run_dir / f"segment_{seg['segment_index']:02d}"
        seg_dir.mkdir(exist_ok=True)
        for j, frame_b64 in enumerate(seg["frames_base64_jpeg"]):
            frame_bytes = base64.b64decode(frame_b64)
            (seg_dir / f"frame_{j:03d}.jpg").write_bytes(frame_bytes)

    print(f"Results saved to {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
async def health():
    """Health-check used by Docker HEALTHCHECK and load balancers.

    Returns 503 while the model is still loading so that orchestrators
    do not route traffic to an unready instance.
    """
    if _model is None:
        return JSONResponse(status_code=503, content={"status": "loading"})
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """Receive an MP4 video clip and return anomaly scores and Grad-CAM maps.

    For each segment the response includes base64-encoded JPEG frames with a
    Grad-CAM heatmap overlay.  Normal segments receive a black (zero) map so
    that the original frame is returned unaltered, while anomalous segments
    show a jet-coloured Grad-CAM highlight indicating which spatiotemporal
    regions drove the anomaly score.

    Response fields
    ---------------
    anomaly_score   float           – max score across all segments
    segment_scores  list[float]     – per-segment MIL scores
    is_anomalous    bool            – True when anomaly_score > threshold
    threshold       float           – value of ANOMALY_THRESHOLD at serve time
    num_segments    int             – number of temporal segments evaluated
    segments        list[dict]      – one entry per segment:

        segment_index       int       – position in segment_scores
        segment_score       float     – anomaly score
        is_anomalous        bool      – True when score > threshold
        frames_base64_jpeg  list[str] – base64 JPEG frames; normal segments
                                        show the plain frame (black map),
                                        anomalous segments show Grad-CAM overlay
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model is still loading.")

    if file.filename and not file.filename.lower().endswith(".mp4"):
        raise HTTPException(status_code=400, detail="Only MP4 files are accepted.")

    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        video_tensor = _preprocess_video(tmp_path)  # (1, N, T, C, H, W)

        # Decode raw uint8 frames for heatmap overlay.
        raw_all, _, _ = torchvision.io.read_video(
            tmp_path, output_format="TCHW", pts_unit="sec"
        )  # (total_T, C, H_orig, W_orig)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        if tmp_path:
            Path(tmp_path).unlink(missing_ok=True)

    async with _model_lock:
        video_tensor_device = video_tensor.to(_device)
        with torch.no_grad():
            scores = _model(video_tensor_device)  # (1, N)

        segment_scores: list[float] = scores.squeeze(0).tolist()
        anomaly_score = float(max(segment_scores))

        total_frames = raw_all.shape[0]
        start_indices = torch.linspace(0, max(total_frames - 1, 0), NUM_SEGMENTS).long()
        resize_to = _preproc["resize_to"]

        segments_out: list[dict] = []
        raw_segments_list: list[torch.Tensor] = []
        cam_list: list[torch.Tensor] = []
        for seg_idx, seg_score in enumerate(segment_scores):
            seg_is_anomalous = seg_score > ANOMALY_THRESHOLD

            # ---- Extract & normalise raw frames for this segment ----
            s = start_indices[seg_idx].item()
            e = min(s + FRAMES_PER_SEGMENT, total_frames)
            seg_raw = raw_all[s:e]  # (T', C, H_orig, W_orig) uint8

            if seg_raw.shape[-2:] != tuple(resize_to):
                seg_raw = F.interpolate(
                    seg_raw.float(),
                    size=resize_to,
                    mode="bilinear",
                    align_corners=False,
                ).byte()

            if seg_raw.shape[0] < FRAMES_PER_SEGMENT:
                if seg_raw.shape[0] > 0:
                    pad = seg_raw[-1:].expand(
                        FRAMES_PER_SEGMENT - seg_raw.shape[0], -1, -1, -1
                    )
                    seg_raw = torch.cat([seg_raw, pad], dim=0)
                else:
                    seg_raw = torch.zeros(
                        (FRAMES_PER_SEGMENT, 3, *resize_to), dtype=torch.uint8
                    )

            # ---- Grad-CAM or black map ----
            if seg_is_anomalous:
                seg_tensor = video_tensor_device[:, seg_idx, :]  # (1, T, C, H, W)
                cam = _gradcam_for_segment(seg_tensor)  # (T, H, W) in [0,1]
            else:
                cam = torch.zeros(
                    FRAMES_PER_SEGMENT, *resize_to
                )  # black — no activation shown

            raw_segments_list.append(seg_raw)
            cam_list.append(cam)
            segments_out.append(
                {
                    "segment_index": seg_idx,
                    "segment_score": seg_score,
                    "is_anomalous": seg_is_anomalous,
                    "frames_base64_jpeg": _overlay_heatmap(seg_raw, cam),
                }
            )

    result = {
        "anomaly_score": anomaly_score,
        "segment_scores": segment_scores,
        "is_anomalous": anomaly_score > ANOMALY_THRESHOLD,
        "threshold": ANOMALY_THRESHOLD,
        "num_segments": NUM_SEGMENTS,
        "segments": segments_out,
    }

    if SAVE_LOCALLY:
        run_dir = _save_results_locally(file.filename or "upload.mp4", result)
        _create_result_video(run_dir, result, raw_segments_list, cam_list)

    return result


@app.post("/admin/reload")
async def reload_model(
    model_name: str | None = None,
    model_alias: str | None = None,
):
    """Reload model and preprocessing from MLflow without restarting the app.

    Optional query parameter:
    * ``model_name`` — if provided, overrides the configured model name.
    * ``model_alias`` — if provided, overrides the configured model alias.
    """
    target_model_name = model_name or MODEL_NAME
    target_model_alias = model_alias or MODEL_ALIAS

    async with _model_lock:
        try:
            _reload_model_from_registry(target_model_name, target_model_alias)
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Model reload failed: {exc}",
            ) from exc

    return {
        "status": "reloaded",
        "model_name": MODEL_NAME,
        "model_alias": MODEL_ALIAS,
        "device": str(_device),
        "num_segments": NUM_SEGMENTS,
        "frames_per_segment": FRAMES_PER_SEGMENT,
        "threshold": ANOMALY_THRESHOLD,
    }

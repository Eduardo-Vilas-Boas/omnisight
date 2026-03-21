# OmniSight — Industrial Video Anomaly Detection

A video-based anomaly detection system for industrial environments, built as a portfolio project.
It combines a pretrained **VideoMAE** backbone with a **Multiple Instance Learning (MIL)** ranking
head, trained end-to-end on the [IPAD dataset](https://arxiv.org/pdf/2404.15033).

---

## Motivation

Industrial anomaly detection is a high-value problem: a system that can flag abnormal events from
raw camera footage — without needing frame-level annotations — significantly reduces the cost of
labelling and makes it practical to add new inspection categories over time.

This project explores two key ideas:

1. **Weak supervision via MIL** — Although the IPAD dataset provides frame-level labels, using
   only video-level labels (normal / abnormal) makes it far easier to annotate new data — a
   user only needs to tag an entire clip, not individual frames. The
   [Sultani et al. (2018)](https://arxiv.org/pdf/1801.04264) MIL ranking loss is designed for
   exactly this setting: each video is treated as a *bag* of segments, and the model is trained
   so that the highest-scoring segment of an abnormal video outscores the highest-scoring segment
   of a normal one. This means extending the system to a new category only requires video-level
   labels.

2. **VideoMAE as a frozen feature extractor** — Rather than training a video backbone from
   scratch, the model uses
   [`MCG-NJU/videomae-small-finetuned-ssv2`](https://huggingface.co/MCG-NJU/videomae-base) as a
   frozen foundation model. VideoMAE is a masked-autoencoder pretrained on large video corpora,
   so its representations transfer well to new domains with minimal fine-tuning, cutting both
   training time and compute requirements.

---

## Architecture

```
Input video (B, N segments, T frames, C, H, W)
        │
        ▼
┌───────────────────────────────┐
│  VideoMAE (frozen backbone)   │  MCG-NJU/videomae-small-finetuned-ssv2
│  Global-average-pool CLS token│  hidden dim = 384
└───────────────────────────────┘
        │  features (B·N, 384)
        ▼
┌───────────────────────────────┐
│  MIL Ranking Head             │
│  Linear(384→256) → ReLU       │
│  Dropout(0.6)                 │
│  Linear(256→1)                │
└───────────────────────────────┘
        │  segment scores (B, N)
        ▼
   MIL Ranking Loss
   (ranking + smoothness)
```

### MIL intuition

With only video-level labels, we cannot know *which* segment inside an abnormal clip actually
contains the anomaly — only that *at least one* does. The key insight from
[Sultani et al. (2018)](https://arxiv.org/pdf/1801.04264) is that we don't need to know: we just
require that **the highest-scoring segment of any abnormal video scores higher than the
highest-scoring segment of any normal video**. Normal videos are anomaly-free by definition, so
every segment in them should score low. If the model learns this ranking, the top segment of an
abnormal bag has implicitly been identified as the most suspicious one.

### Loss function

Adapted from [Sultani et al., CVPR 2018](https://arxiv.org/pdf/1801.04264):

```
L = mean( max(0, 1 - max_score(abnormal_bag) + max_score(normal_bag)) )
  + λ * mean( (score_t - score_{t+1})² )
```

The ranking term enforces the margin described above. The smoothness term penalises large
score swings between consecutive segments, discouraging the model from spiking on a single
frame and ignoring the rest.

---

## Dataset

**IPAD — Industrial Process Anomaly Detection**
([paper](https://arxiv.org/pdf/2404.15033) · [project page](https://ljf1113.github.io/IPAD_VAD/))

> Liu, J. et al. (2024). *IPAD: Industrial Process Anomaly Detection Dataset*. Shanghai Jiao Tong University & Lenovo Research.

<video src="https://ljf1113.github.io/IPAD_VAD/static/videos/IPAD_intro.mp4" controls width="100%">
  <a href="https://ljf1113.github.io/IPAD_VAD/static/videos/IPAD_intro.mp4">Watch the IPAD dataset introduction video</a>
</video>

### Overview

IPAD is the first large-scale video anomaly detection dataset specifically designed for
**industrial production scenarios**. Key properties:

- **16 industrial devices** across both real-world factory footage and synthetic simulations
- **6+ hours** of video in total
- **Two splits** by recording type:
  - `R01`–`R04` — real-world recordings captured on-site at factories
  - `S01`–`S12` — synthetic sequences generated to cover a wider range of anomaly types
- **Frame-level labels** — every frame is annotated as normal or anomalous, enabling precise
  evaluation (though this project intentionally uses only the video-level label during training
  — see [Motivation](#motivation))
- **Periodicity annotations** — the dataset also labels the periodic structure of each
  industrial process, which is a distinctive trait of repetitive manufacturing operations

The categories were selected through on-site factory research in collaboration with process
engineers, making the anomaly types representative of real deployment scenarios.

### Downloading the dataset

1. Go to the dataset page: https://ljf1113.github.io/IPAD_VAD/
2. Follow the download instructions there to obtain the dataset archive(s).
3. Create a `data/` folder at the root of this repo and unzip the archive there:

```bash
mkdir -p data
# move the downloaded archive(s) into data/ then:
unzip IPAD_dataset.zip -d data/
```

The expected layout after extraction:

```
data/
  IPAD_dataset/
    {category}/
      training/frames/{seq}/   ← anomaly-free sequences
      testing/frames/{seq}/    ← may contain anomalies
      test_label/{seq}.npy     ← per-sequence binary label array
```

The training script automatically splits and copies the data into a processed layout alongside
the raw data:

```
data/
  IPAD_dataset_processed/
    {category}/
      train/{normal,abnormal}/
      val/{normal,abnormal}/
      test/{normal,abnormal}/
```

---

## Project structure

```
omnisight/
├── src/
│   ├── anomaly_detector/
│   │   ├── model.py          # VideoAnomalyDetector (Lightning module)
│   │   └── dataset.py        # VideoAnomalyDataset
│   ├── train_model.py        # Hydra entry-point: data prep + training + MLflow logging
│   ├── app.py                # FastAPI serving endpoint
│   └── utils.py
├── conf/
│   └── train.yaml            # Hydra config (dataset, trainer, MLflow)
├── Dockerfile.training       # GPU training image (CUDA + uv)
├── Dockerfile.serving        # Inference API image
├── docker-compose.yml        # Compose stack: PostgreSQL + MLflow + serving (+ optional training)
└── pyproject.toml
```

---

## Quick start

### Local development

**Prerequisites:** [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12.

```bash
uv sync --dev
source .venv/bin/activate
```

**Run training** (adjust `dataset.category` and paths as needed):

```bash
python -m src.train_model \
    dataset.root=./data/IPAD_dataset \
    dataset.category=R03 \
    training.num_gpus=1
```

Experiment metrics, parameters, and the best checkpoint are automatically logged to MLflow.

---

### Docker

The full stack (PostgreSQL, MLflow tracking server, and the serving API) is managed with Docker
Compose:

```bash
# Start infrastructure + serving API
docker compose up -d

# Run training (on-demand, requires NVIDIA Container Toolkit)
docker compose --profile train up training
```

MLflow UI is available at `http://localhost:5000`.
The inference API is available at `http://localhost:8000`.

---

## Configuration

All training parameters are controlled via `conf/train.yaml` and can be overridden on the
command line (Hydra):

| Key | Default | Description |
|-----|---------|-------------|
| `dataset.category` | `R03` | IPAD category to train on |
| `dataset.num_segments` | `16` | Segments per video bag |
| `dataset.frames_per_segment` | `16` | Frames per segment |
| `training.lr` | `1e-4` | Learning rate |
| `training.batch_size` | `8` | Per-step batch size |
| `training.effective_batch_size` | `8` | Target batch size (gradient accumulation) |
| `training.max_epochs` | `50` | Training epochs (matches the IPAD paper's protocol) |
| `training.precision` | `16-mixed` | Mixed-precision mode |
| `mlflow.tracking_uri` | `http://localhost:5000` | MLflow backend |

---

## Evaluation

The model is evaluated on a held-out test split after training. Metrics logged to MLflow:

- **AUROC** — area under the ROC curve for bag-level anomaly scores
- **Best F1** — F1 score at the threshold that maximises it (swept over 200 candidates)

---

## Tech stack

| Component | Library / Tool |
|-----------|---------------|
| Model | [VideoMAE](https://huggingface.co/MCG-NJU/videomae-base) via HuggingFace Transformers |
| Training loop | PyTorch Lightning |
| Config | Hydra + OmegaConf |
| Experiment tracking | MLflow (PostgreSQL backend) |
| Packaging | uv |
| Containerisation | Docker + NVIDIA Container Toolkit |
| Inference API | FastAPI |

---

## References

- Sultani, W., Chen, C., & Shah, M. (2018). *Real-world Anomaly Detection in Surveillance Videos*. CVPR. https://arxiv.org/pdf/1801.04264
- Wang, L. et al. (2024). *IPAD: Industrial Process Anomaly Detection Dataset*. https://arxiv.org/pdf/2404.15033
- Tong, Z. et al. (2022). *VideoMAE: Masked Autoencoders are Data-Efficient Learners for Self-Supervised Video Pre-Training*. NeurIPS. https://huggingface.co/MCG-NJU/videomae-base

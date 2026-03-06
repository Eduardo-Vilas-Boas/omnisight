from __future__ import annotations

import getpass
import os
from datetime import datetime
from pathlib import Path
import shutil
import hydra
import mlflow
import mlflow.pytorch
from matplotlib import category
import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from omnisight.ingest.anomaly_detector.dataset import VideoAnomalyDataset
from omnisight.ingest.anomaly_detector.model import VideoAnomalyDetector


def process_dataset(
    dataset_root: Path,
    category: str,
    processed_dataset_root: Path,
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> None:
    """Process raw IPAD frames into train/val/test splits with normal and abnormal sub-folders.

    Output layout::

        processed_dataset_root/
          {category}/
            train/
              normal/
              abnormal/
            val/
              normal/
              abnormal/
            test/
              normal/
              abnormal/
    """
    category_dataset_root = dataset_root / category

    if not category_dataset_root.exists():
        raise FileNotFoundError(f"Category folder not found: {category_dataset_root}")

    # ------------------------------------------------------------------ #
    # 1. Collect sources for each class                                   #
    # ------------------------------------------------------------------ #
    normal_sources: list[tuple[Path, str]] = []  # (src_path, dest_name)
    abnormal_sources: list[tuple[Path, str]] = []

    # All training sequences are anomaly-free → normal
    training_base = category_dataset_root / "training" / "frames"
    for seq in sorted(os.listdir(training_base)):
        normal_sources.append((training_base / seq, f"training_{seq}"))

    # Testing sequences are classified by their label file
    testing_base = category_dataset_root / "testing" / "frames"
    for seq in sorted(os.listdir(testing_base)):
        label_path = category_dataset_root / "test_label" / f"{int(seq):03}.npy"
        if not label_path.exists():
            raise RuntimeError(f"Label file not found: expected at {label_path}")
        label = np.load(label_path).astype(np.float32)
        src = testing_base / seq
        if label.sum() == 0:
            normal_sources.append((src, f"testing_{seq}"))
        else:
            abnormal_sources.append((src, f"testing_{seq}"))

    # ------------------------------------------------------------------ #
    # 2. Split each class into train / val / test (folder-level)         #
    # ------------------------------------------------------------------ #
    def _split(
        sources: list[tuple[Path, str]], val_frac: float, test_frac: float
    ) -> tuple[list, list, list]:
        n_total = len(sources)
        n_test = max(1, round(n_total * test_frac))
        n_val = max(1, round(n_total * val_frac))

        test_set = sources[:n_test]
        val_set = sources[n_test : n_test + n_val]
        train_set = sources[n_test + n_val :]
        return train_set, val_set, test_set

    train_normal, val_normal, test_normal = _split(
        normal_sources, val_split, test_split
    )
    train_abnormal, val_abnormal, test_abnormal = _split(
        abnormal_sources, val_split, test_split
    )

    # ------------------------------------------------------------------ #
    # 3. Copy to destination folders                                       #
    # ------------------------------------------------------------------ #
    splits: dict[str, dict[str, list]] = {
        "train": {"normal": train_normal, "abnormal": train_abnormal},
        "val": {"normal": val_normal, "abnormal": val_abnormal},
        "test": {"normal": test_normal, "abnormal": test_abnormal},
    }
    for split_name, classes in splits.items():
        for class_name, sources in classes.items():
            dest_base = processed_dataset_root / category / split_name / class_name
            for src, dest_name in sources:
                shutil.copytree(src, dest_base / dest_name)


def _run_training(cfg: DictConfig) -> None:
    torch.manual_seed(42)
    np.random.seed(42)

    selected_dataset_root = Path(cfg.dataset.root)
    process_dataset_root = (
        selected_dataset_root.parent / f"{selected_dataset_root.name}_processed"
    )
    print(f"Using dataset source: local at {selected_dataset_root}")

    if not selected_dataset_root.exists():
        raise FileNotFoundError(f"Training folder not found: {selected_dataset_root}")

    Path(cfg.output.dir).mkdir(parents=True, exist_ok=True)

    categories = [p.name for p in selected_dataset_root.iterdir() if p.is_dir()]
    print(f"Found categories: {categories}")

    if not os.path.exists(process_dataset_root):
        process_dataset(
            selected_dataset_root,
            cfg.dataset.category,
            process_dataset_root,
            val_split=cfg.dataset.val_split,
            test_split=cfg.dataset.test_split,
        )

    print(f"Training model for category '{cfg.dataset.category}'...")
    print(
        f"Dataset for category '{cfg.dataset.category}' has been processed and is located at: {process_dataset_root / cfg.dataset.category}"
    )

    model = VideoAnomalyDetector(lr=cfg.training.lr)

    # Initialize Datasets
    # Consider lowering num_segments (e.g., 16) if memory is still an issue
    num_segments = cfg.dataset.get("num_segments", 32)
    frames_per_segment = cfg.dataset.get("frames_per_segment", 16)

    category_processed_root = process_dataset_root / cfg.dataset.category
    train_dataset = VideoAnomalyDataset(
        root_dir=str(category_processed_root / "train"),
        num_segments=num_segments,
        frames_per_segment=frames_per_segment,
    )
    val_dataset = VideoAnomalyDataset(
        root_dir=str(category_processed_root / "val"),
        num_segments=num_segments,
        frames_per_segment=frames_per_segment,
    )
    test_dataset = VideoAnomalyDataset(
        root_dir=str(category_processed_root / "test"),
        num_segments=num_segments,
        frames_per_segment=frames_per_segment,
    )

    num_workers = cfg.training.get("num_workers", 1)  # Default mapped to a safer 2

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    # 2. Setup Checkpointing
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath="checkpoints/",
        filename="videomae-anomaly-{epoch:02d}-{val_loss:.4f}",
        save_top_k=1,
        mode="min",
    )

    # 3. Resolve accelerator and devices
    if cfg.training.num_gpus == 0 or not torch.cuda.is_available():
        accelerator, devices = "cpu", "auto"
    else:
        accelerator = "gpu"
        devices = cfg.training.num_gpus  # -1 = all available GPUs on the node

    if cfg.training.effective_batch_size % cfg.training.batch_size != 0:
        raise ValueError(
            f"Effective batch size ({cfg.training.effective_batch_size}) must be divisible by batch size ({cfg.training.batch_size})"
        )

    accumulate_grad_batches = int(
        cfg.training.effective_batch_size / cfg.training.batch_size
    )

    model_name = (
        cfg.mlflow.get("register_model_name")
        or f"{cfg.mlflow.experiment_name}-{cfg.dataset.category}"
    )

    # 4. Open the MLflow run explicitly so it stays active through fit → test → registration
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    with mlflow.start_run(
        run_name=f"{cfg.mlflow.run_name}_{cfg.dataset.category}"
    ) as run:

        # 5. Trainer — attach to the already-open run via run_id
        trainer = pl.Trainer(
            max_epochs=cfg.training.max_epochs,
            accumulate_grad_batches=accumulate_grad_batches,
            accelerator=accelerator,
            devices=devices,
            precision=cfg.training.precision,
            sync_batchnorm=True,
            callbacks=[checkpoint_callback],
            logger=pl.loggers.MLFlowLogger(
                tracking_uri=cfg.mlflow.tracking_uri,
                experiment_name=cfg.mlflow.experiment_name,
                run_id=run.info.run_id,
            ),
        )

        trainer.fit(model, train_dataloader, val_dataloader)

        trainer.test(model, test_dataloader)

        # ------------------------------------------------------------------ #
        # Register the best model in the MLflow Model Registry               #
        # ------------------------------------------------------------------ #
        if checkpoint_callback.best_model_path and trainer.is_global_zero:
            print("Training completed!")
            print(f"Best model saved at: {checkpoint_callback.best_model_path}")
            print(f"Best validation loss: {checkpoint_callback.best_model_score}")

            print("Loading and logging best model to MLflow...")

            best_model = VideoAnomalyDetector.load_from_checkpoint(
                checkpoint_path=checkpoint_callback.best_model_path
            )
            best_model.to(device="cpu")
            best_model.eval()

            # Retrieve test metrics written by on_test_epoch_end via self.log()
            test_auroc = float(trainer.callback_metrics.get("test_auroc", 0.0))
            test_best_f1 = float(trainer.callback_metrics.get("test_best_f1", 0.0))
            test_best_threshold = float(
                trainer.callback_metrics.get("test_best_threshold", 0.0)
            )

            model_metadata = {
                "model_architecture": "VideoAnomalyDetector",
                "best_val_loss": float(checkpoint_callback.best_model_score),
                "test_auroc": test_auroc,
                "test_best_f1": test_best_f1,
                "test_best_threshold": test_best_threshold,
                "category": cfg.dataset.category,
                "checkpoint_path": checkpoint_callback.best_model_path,
                "num_segments": num_segments,
                "frames_per_segment": frames_per_segment,
            }

            # Run is already active — log directly, no start_run needed
            mlflow.log_artifact(checkpoint_callback.best_model_path, "model_checkpoint")

            registered_model = mlflow.pytorch.log_model(
                pytorch_model=best_model,
                artifact_path=model_name,
                metadata=model_metadata,
            )

            model_version = mlflow.register_model(
                model_uri=registered_model.model_uri,
                name=model_name,
            )

            client = mlflow.MlflowClient()
            created_by = getpass.getuser()
            creation_timestamp = datetime.now().isoformat()
            val_loss_str = f"{float(checkpoint_callback.best_model_score):.4f}"

            # Tags on the registered model (shown in the Registry overview)
            client.set_registered_model_tag(
                name=model_name, key="category", value=cfg.dataset.category
            )
            client.set_registered_model_tag(
                name=model_name, key="experiment_name", value=cfg.mlflow.experiment_name
            )
            client.set_registered_model_tag(
                name=model_name, key="test_auroc", value=f"{test_auroc:.4f}"
            )
            client.set_registered_model_tag(
                name=model_name, key="test_best_f1", value=f"{test_best_f1:.4f}"
            )

            # Tags on the specific model version (shown when clicking the version)
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="category",
                value=cfg.dataset.category,
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="val_loss",
                value=val_loss_str,
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="test_auroc",
                value=f"{test_auroc:.4f}",
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="test_best_f1",
                value=f"{test_best_f1:.4f}",
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="test_best_threshold",
                value=f"{test_best_threshold:.4f}",
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="model_type",
                value="VideoAnomalyDetector",
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="created_by",
                value=created_by,
            )
            client.set_model_version_tag(
                name=model_name,
                version=model_version.version,
                key="creation_date",
                value=creation_timestamp.split("T")[0],
            )

            print(
                f"Model registered in MLflow Model Registry: {model_name} v{model_version.version}"
            )
            print(f"  - Category:            {cfg.dataset.category}")
            print(f"  - Created by:          {created_by}")
            print(f"  - Best Validation Loss: {val_loss_str}")
            print(f"  - Test AUROC:          {test_auroc:.4f}")
            print(f"  - Test Best F1:        {test_best_f1:.4f}")
            print(f"Model logged with run ID: {run.info.run_id}")
            print(
                f"To load registered model: mlflow.pytorch.load_model('models:/{model_name}/{model_version.version}')"
            )

    print(f"Experiment Results for category '{cfg.dataset.category}':")


# config_path is relative to this source file; points to <project_root>/conf
@hydra.main(version_base=None, config_path="../../../../conf", config_name="train")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    _run_training(cfg)


if __name__ == "__main__":
    main()

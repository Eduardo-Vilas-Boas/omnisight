import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics.functional.classification import binary_auroc
from transformers import VideoMAEModel


class VideoAnomalyDetector(L.LightningModule):
    def __init__(self, lr=1e-5, weight_decay=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # 1. Feature Extractor (Hugging Face VideoMAE)
        self.backbone = VideoMAEModel.from_pretrained(
            "MCG-NJU/videomae-small-finetuned-ssv2"
        )
        # Freeze backbone to focus on the MIL head initially
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        # 2. MIL Ranking Head
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x shape: (batch, num_segments, T, C, H, W)
        b, n, t, c, h, w = x.shape
        flat_x = x.view(-1, t, c, h, w)  # (B*N, T, C, H, W) as VideoMAE expects

        # Extract features (using the [CLS] token equivalent in VideoMAE)
        with torch.no_grad():  # backbone is frozen — no need to store activations
            # Process in smaller chunks to avoid OOM
            chunk_size = 4
            all_features = []
            for i in range(0, flat_x.size(0), chunk_size):
                chunk = flat_x[i : i + chunk_size]
                outputs = self.backbone(pixel_values=chunk)
                chunk_features = outputs.last_hidden_state.mean(
                    dim=1
                )  # Global average pooling
                all_features.append(chunk_features)
            features = torch.cat(all_features, dim=0)

        scores = self.classifier(features)
        return scores.view(b, n)  # (batch, num_segments)

    # Inspired MIL loss function from https://arxiv.org/pdf/1801.04264
    # anomalous_scores: (B, N), normal_scores: (B, N)
    # The dataset guarantees each item is a matched pair, so no label-based splitting needed.
    def mil_loss(self, anomalous_scores, normal_scores):
        max_a = torch.max(anomalous_scores, dim=1)[0]  # (B,)
        max_n = torch.max(normal_scores, dim=1)[0]  # (B,)

        # Pairwise margin ranking loss: every anomalous bag must outscore every normal bag
        ranking_loss = F.relu(1.0 - max_a.unsqueeze(1) + max_n.unsqueeze(0)).mean()

        # Sparsity & Smoothness constraints on anomalous bags
        smoothness = torch.mean(
            (anomalous_scores[:, :-1] - anomalous_scores[:, 1:]) ** 2
        )

        # Normally, the anomalous scores should be sparse
        # But the current dataset video segments are too short,
        # so we just ignore this term for now to avoid over-penalising the model.
        # # sparsity = torch.mean(anomalous_scores)

        return ranking_loss + (8e-5 * smoothness)  # + (8e-5 * sparsity)

    def training_step(self, batch, batch_idx):
        # batch is a dict with "normal" and "abnormal" keys, each (B, N, T, C, H, W)
        normal_scores = self(batch["normal"])  # (B, N)
        anomalous_scores = self(batch["abnormal"])  # (B, N)
        loss = self.mil_loss(anomalous_scores, normal_scores)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        normal_scores = self(batch["normal"])  # (B, N)
        anomalous_scores = self(batch["abnormal"])  # (B, N)
        loss = self.mil_loss(anomalous_scores, normal_scores)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_start(self):
        self._test_scores: list[torch.Tensor] = []
        self._test_labels: list[torch.Tensor] = []

    def test_step(self, batch, batch_idx):
        normal_scores = self(batch["normal"])  # (B, N)
        anomalous_scores = self(batch["abnormal"])  # (B, N)

        # Use the max-scoring segment as the bag-level score
        max_normal = torch.max(normal_scores, dim=1)[0]  # (B,)
        max_anomalous = torch.max(anomalous_scores, dim=1)[0]  # (B,)

        scores = torch.cat([max_normal, max_anomalous])  # (2B,)
        labels = torch.cat(
            [torch.zeros_like(max_normal), torch.ones_like(max_anomalous)]
        )  # (2B,)

        self._test_scores.append(scores.cpu())
        self._test_labels.append(labels.cpu())

    def on_test_epoch_end(self):

        scores = torch.cat(self._test_scores)  # (N_total,)
        labels = torch.cat(self._test_labels).long()  # (N_total,)

        print(
            f"\nCollected {len(scores)} test samples: {labels.sum().item()} anomalous, {len(labels) - labels.sum().item()} normal"
        )
        print("Sample scores:", scores[:10])
        print("Sample labels:", labels[:10])

        # Binary AUROC
        auroc = binary_auroc(scores, labels)
        self.log("test_auroc", auroc, prog_bar=True)

        # Sweep thresholds to find the one that maximises F1
        thresholds = torch.linspace(scores.min().item(), scores.max().item(), 200)
        best_f1, best_threshold = 0.0, thresholds[0].item()
        labels_f = labels.float()
        for t in thresholds:
            preds = (scores >= t).float()
            tp = (preds * labels_f).sum()
            fp = (preds * (1.0 - labels_f)).sum()
            fn = ((1.0 - preds) * labels_f).sum()
            f1 = (2.0 * tp / (2.0 * tp + fp + fn + 1e-8)).item()
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = t.item()

        self.log("test_best_f1", torch.tensor(best_f1), prog_bar=True)
        self.log("test_best_threshold", torch.tensor(best_threshold))
        print(
            f"\nTest AUROC: {auroc:.4f} "
            f"| Best F1: {best_f1:.4f} at threshold={best_threshold:.4f}"
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchvision.io
from transformers import VideoMAEImageProcessor


class VideoAnomalyDataset(Dataset):
    def __init__(self, root_dir, num_segments=32, frames_per_segment=16):
        self.root_dir = root_dir

        good_category_dir = os.path.join(root_dir, "normal")
        bad_category_dir = os.path.join(root_dir, "abnormal")

        self.good_video_dirs = [
            os.path.join(good_category_dir, d, "")
            for d in sorted(os.listdir(good_category_dir))
            if os.path.isdir(os.path.join(good_category_dir, d))
        ]

        self.bad_video_dirs = [
            os.path.join(bad_category_dir, d, "")
            for d in sorted(os.listdir(bad_category_dir))
            if os.path.isdir(os.path.join(bad_category_dir, d))
        ]

        if len(self.good_video_dirs) == 0 or len(self.bad_video_dirs) == 0:
            raise RuntimeError(
                f"Both good and bad video directories must be present and non-empty in {root_dir}"
            )

        self.good_bad_combinations = [
            {"good": x1, "abnormal": x2}
            for x1 in self.good_video_dirs
            for x2 in self.bad_video_dirs
        ]

        self.num_segments = num_segments
        self.frames_per_segment = frames_per_segment

        image_processor = VideoMAEImageProcessor.from_pretrained(
            "MCG-NJU/videomae-base"
        )

        mean = image_processor.image_mean
        std = image_processor.image_std
        # Store as (C, 1, 1) tensors for broadcasting over (C, H, W) clips
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        if "shortest_edge" in image_processor.size:
            height = width = image_processor.size["shortest_edge"]
        else:
            height = image_processor.size["height"]
            width = image_processor.size["width"]
        self.resize_to = (height, width)

    def __len__(self):
        return len(self.good_bad_combinations)

    def get_video_segments(self, video_folder):

        image_paths = sorted(
            [
                os.path.join(video_folder, x)
                for x in os.listdir(video_folder)
                if x.endswith(".jpg")
            ]
        )

        total_frames = len(image_paths)
        # Random temporal segment sampling: divide the video into num_segments
        # equal bins and pick a random start within each bin.  This ensures
        # full temporal coverage while exposing different clips every epoch.
        if total_frames <= self.num_segments:
            # Fewer frames than segments — fall back to evenly spaced.
            start_indices = (
                torch.linspace(0, max(total_frames - 1, 0), self.num_segments)
                .long()
                .tolist()
            )
        else:
            bin_size = total_frames // self.num_segments
            start_indices = [
                torch.randint(i * bin_size, (i + 1) * bin_size, (1,)).item()
                for i in range(self.num_segments)
            ]

        video_data = []
        for start in start_indices:
            end = min(start + self.frames_per_segment, total_frames)

            clip_frames = []
            for i in range(start, end):
                frame = torchvision.io.read_image(
                    str(image_paths[i])
                )  # (C, H, W) uint8

                # Resize directly to final size in one step to avoid a wasteful
                # intermediate 256x256 allocation
                frame = (
                    F.interpolate(
                        frame.unsqueeze(0).float(),
                        size=self.resize_to,
                        mode="bilinear",
                        align_corners=False,
                    )
                    .squeeze(0)
                    .byte()
                )  # (C, H, W) uint8
                clip_frames.append(frame)

            if len(clip_frames) > 0:
                clip = torch.stack(clip_frames)  # (T, C, H, W)
            else:
                clip = torch.zeros((0, 3, *self.resize_to), dtype=torch.uint8)

            # Pad with the last frame if the clip is shorter than expected
            if clip.shape[0] < self.frames_per_segment:
                if clip.shape[0] > 0:
                    pad = clip[-1:].expand(
                        self.frames_per_segment - clip.shape[0], -1, -1, -1
                    )
                    clip = torch.cat([clip, pad], dim=0)
                else:  # Fallback if purely empty
                    clip = torch.zeros(
                        (self.frames_per_segment, 3, *self.resize_to), dtype=torch.uint8
                    )

            # Normalize to [0, 1] then apply ImageNet mean/std to match VideoMAE pretraining
            clip = clip.float() / 255.0
            clip = (clip - self.mean) / self.std

            video_data.append(clip)

        return torch.stack(video_data)

    def __getitem__(self, idx):
        # Read all frames: returns (T, H, W, C) uint8 tensor

        return {
            "normal": self.get_video_segments(self.good_bad_combinations[idx]["good"]),
            "abnormal": self.get_video_segments(
                self.good_bad_combinations[idx]["abnormal"]
            ),
        }

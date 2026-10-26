"""Utilities for constructing OTFS evidence atoms.

Heavy model dependencies are imported lazily so the OTFS optimization core can
be installed and tested with NumPy alone.
"""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import Callable, Sequence

import numpy as np


def one_fps_indices(total_frames: int, fps: float) -> list[int]:
    """Return the exact 1-fps candidate indices used in the paper experiments."""

    if total_frames <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError("invalid video length or fps")
    stride = max(1, int(fps))
    count = max(1, int(total_frames / stride))
    return [i * stride for i in range(count)]


def adjacent_cosine_distance(features: np.ndarray) -> np.ndarray:
    """Compute transition evidence with zero assigned to the first frame."""

    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) == 0:
        raise ValueError("features must have shape [frames, dimensions]")
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    normalized = np.divide(x, norm, out=np.zeros_like(x), where=norm > 0)
    result = np.zeros(len(x), dtype=np.float64)
    if len(x) > 1:
        result[1:] = 1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)
    return result


def sparse_reward_interpolator(
    boundaries: Sequence[int], sampled_indices: Sequence[int], rewards: Sequence[float]
) -> Callable[[int], float]:
    """Build the linear interpolator used to densify DIG utility rewards."""

    if not boundaries or not sampled_indices or not rewards:
        return lambda _: 0.0
    if len(sampled_indices) != len(rewards):
        raise ValueError("sampled_indices and rewards have different lengths")
    if len(boundaries) != len(sampled_indices) + 1:
        raise ValueError("DIG boundaries must contain one more entry than rewards")
    points = [(int(boundaries[0]), 0.0)]
    points.extend((int(i), float(r)) for i, r in zip(sampled_indices, rewards))
    points.append((int(boundaries[-1]), 0.0))
    points.sort()
    deduplicated: list[tuple[int, float]] = []
    for index, reward in points:
        if deduplicated and deduplicated[-1][0] == index:
            deduplicated[-1] = (index, max(deduplicated[-1][1], reward))
        else:
            deduplicated.append((index, reward))
    xs = [point[0] for point in deduplicated]
    ys = [point[1] for point in deduplicated]

    def interpolate(frame_index: int) -> float:
        if frame_index <= xs[0] or frame_index >= xs[-1]:
            return 0.0
        right = bisect_right(xs, frame_index)
        x0, y0 = xs[right - 1], ys[right - 1]
        x1, y1 = xs[right], ys[right]
        if frame_index == x0:
            return float(y0)
        if x1 == x0:
            return float(max(y0, y1))
        alpha = (frame_index - x0) / (x1 - x0)
        return float(y0 + alpha * (y1 - y0))

    return interpolate


def densify_utility(
    frame_indices: Sequence[int],
    boundaries: Sequence[int],
    sampled_indices: Sequence[int],
    rewards: Sequence[float],
) -> list[float]:
    interpolate = sparse_reward_interpolator(boundaries, sampled_indices, rewards)
    return [interpolate(int(index)) for index in frame_indices]


class Dinov2TransitionExtractor:
    """Reuse one DINOv2 model and processor across multiple videos."""

    def __init__(
        self,
        *,
        model_name: str = "facebook/dinov2-base",
        device: str = "cuda",
    ) -> None:
        from transformers import AutoImageProcessor, Dinov2Model

        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = Dinov2Model.from_pretrained(model_name).to(device).eval()

    def extract(
        self, video_path: str | Path, *, batch_size: int = 64
    ) -> tuple[list[int], list[float]]:
        """Extract the paper's 1-fps adjacent-frame transition scores."""

        import torch
        from decord import VideoReader, cpu
        from PIL import Image

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reader = VideoReader(str(video_path), ctx=cpu(0), num_threads=4)
        indices = one_fps_indices(len(reader), float(reader.get_avg_fps()))
        feature_batches = []
        with torch.inference_mode():
            for start in range(0, len(indices), batch_size):
                frames = reader.get_batch(indices[start : start + batch_size]).asnumpy()
                images = [Image.fromarray(frame) for frame in frames]
                inputs = self.processor(images, return_tensors="pt").to(self.device)
                feature_batches.append(
                    self.model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
                )
        features = np.concatenate(feature_batches, axis=0)
        return indices, adjacent_cosine_distance(features).tolist()


def extract_dinov2_transition(
    video_path: str | Path,
    *,
    model_name: str = "facebook/dinov2-base",
    batch_size: int = 64,
    device: str = "cuda",
) -> tuple[list[int], list[float]]:
    """Extract one video; use Dinov2TransitionExtractor for multiple videos."""

    extractor = Dinov2TransitionExtractor(model_name=model_name, device=device)
    return extractor.extract(video_path, batch_size=batch_size)

"""Forward-identical control of feedback into a shared representation."""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


def _alpha(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("feedback alpha must be finite and in [0, 1]")
    return value


class _ScaleFeedback(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: Tensor, alpha: float) -> Tensor:
        ctx.alpha = alpha
        # Return exactly the same values for every alpha, including endpoints.
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: Tensor):
        return gradient * ctx.alpha, None


def scale_feedback(value: Tensor, alpha: float = 1.0) -> Tensor:
    """Identity in the forward pass; multiply only input feedback by alpha.

    Apply this at the INPUT of a derived relation branch. Parameters inside
    that branch keep their ordinary gradients. This scales a raw path-wise
    gradient, not the realized AdamW parameter displacement.
    """
    return _ScaleFeedback.apply(value, _alpha(alpha))


@dataclass(frozen=True)
class FeedbackSchedule:
    """Small explicit policies; updates are one-based optimizer step indices."""

    mode: str = "full"
    alpha: float = 1.0
    stop_after: int = 400

    def __post_init__(self):
        if self.mode not in {"full", "stop", "constant", "late_stop"}:
            raise ValueError("mode must be full, stop, constant, or late_stop")
        _alpha(self.alpha)
        if self.stop_after < 0:
            raise ValueError("stop_after must be nonnegative")

    def __call__(self, update: int) -> float:
        if update < 1:
            raise ValueError("optimizer updates are one-based")
        if self.mode == "full":
            return 1.0
        if self.mode == "stop":
            return 0.0
        if self.mode == "constant":
            return float(self.alpha)
        return 1.0 if update <= self.stop_after else 0.0

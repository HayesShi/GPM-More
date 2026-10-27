"""The existing raw_sum distinct-walk, content-valued phase contraction."""
from __future__ import annotations

import math
import torch
from torch import Tensor


def fixed_frequencies(hidden_dim: int = 256, *, device=None, dtype=None) -> Tensor:
    """Sixteen fixed phases repeated over channels, as in the experiments."""
    return (torch.arange(hidden_dim, device=device, dtype=dtype or torch.float32)
            .remainder(16) * (math.pi / 16))


def closure_phase_pool(paths: Tensor, contextual: Tensor,
                       frequencies: Tensor | None = None) -> Tensor:
    """Return [B,D] sum-phase relations over ordered distinct walks.

    ``paths`` is [B,K,9]; ``contextual`` is [B,K,9,D]. Position zero is
    excluded, while later root revisits remain. For each observed entity,
    the complete per-walk square is subtracted from the all-walk square;
    this excludes ALL same-walk terms, including repeated-position pairs.

    The real contraction is cos(omega*(t+s)), not a relative-position RoPE
    kernel. Only equality of numeric entity IDs is used. No adjacency query
    or additional observation is performed here.
    """
    if paths.ndim != 3 or paths.shape[-1] != 9 or paths.shape[1] < 1:
        raise ValueError("paths must be nonempty [B,K,9] observed walks")
    if contextual.ndim != 4 or contextual.shape[:3] != paths.shape:
        raise ValueError("contextual must have matching [B,K,9,D] shape")
    if paths.device != contextual.device:
        raise ValueError("paths and contextual must be on the same device")
    if paths.dtype not in (torch.int32, torch.int64) or bool((paths < 0).any()):
        raise ValueError("paths must contain nonnegative integer entity IDs")
    batch, count, _ = paths.shape
    if count == 1:
        return contextual.sum(dim=(1, 2)) * 0
    if frequencies is None:
        frequencies = fixed_frequencies(contextual.shape[-1], device=contextual.device,
                                        dtype=contextual.dtype)
    if frequencies.shape != contextual.shape[-1:]:
        raise ValueError("frequencies must have one value per channel")
    ids = paths[..., 1:].long()
    stride = paths.max() + 1
    batch_index = torch.arange(batch, device=paths.device)[:, None, None]
    keys = ids + batch_index * stride
    unique, inverse = torch.unique(keys.flatten(), sorted=True, return_inverse=True)
    entity_batch = torch.div(unique, stride, rounding_mode="floor")
    walk_index = torch.arange(count, device=paths.device)[None, :, None].expand_as(ids)
    local_keys = inverse * count + walk_index.flatten()
    local_unique, local_inverse = torch.unique(local_keys, sorted=True, return_inverse=True)
    local_batch = entity_batch[torch.div(local_unique, count, rounding_mode="floor")]
    omega = frequencies.to(device=contextual.device, dtype=contextual.dtype)
    angles = torch.arange(1, 9, device=contextual.device,
                          dtype=contextual.dtype)[:, None] * omega
    values = contextual[..., 1:, :]
    real = (values * angles.cos()).reshape(-1, values.shape[-1])
    imag = (values * angles.sin()).reshape_as(real)

    def grouped_square(index: Tensor, groups: int) -> Tensor:
        a = values.new_zeros(groups, values.shape[-1]).index_add_(0, index, real)
        b = values.new_zeros(groups, values.shape[-1]).index_add_(0, index, imag)
        return a.square() - b.square()

    shared = grouped_square(inverse, len(unique))
    private = grouped_square(local_inverse, len(local_unique))
    result = values.new_zeros(batch, values.shape[-1]).index_add_(0, entity_batch, shared)
    result = result.index_add_(0, local_batch, -private)
    return result / (count * (count - 1) * 64)

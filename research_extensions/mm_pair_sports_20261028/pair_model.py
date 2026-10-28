"""Matched raw_sum256 LP backbone and sparse same-witness pair contraction.

Research-only implementation; the 1024-dimensional CLIP input is used as one
fused feature vector; no separate modality branches are claimed.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping

import torch
from torch import Tensor, nn

_GPM_SRC = Path(__file__).resolve().parents[2] / "src"
if str(_GPM_SRC) not in sys.path:
    sys.path.insert(0, str(_GPM_SRC))
from gpm_more.model import ModelConfig, RelationalPatternModel


class PairSportsModel(nn.Module):
    """The ordinary and joint arms have identical parameters and initialization.

    Only ``score(..., use_joint=...)`` distinguishes the arms. Contextual node
    states retain their walk depth until the learned low-rank projection. No
    ID embedding is learned: IDs are used solely to join witness identities.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256,
                 rank: int = 16, *, pair_chunk_size: int = 2048,
                 z_layer_norm: bool = True, pattern_dropout: float = 0.1,
                 dropout: float = 0.0, joint_scale_init: float = 0.1):
        super().__init__()
        if rank < 1 or pair_chunk_size < 1:
            raise ValueError("rank and pair_chunk_size must be positive")
        self.hidden_dim = hidden_dim
        self.rank = rank
        self.pair_chunk_size = pair_chunk_size
        self.node_model = RelationalPatternModel(ModelConfig(
            input_dim=input_dim, num_outputs=1, hidden_dim=hidden_dim,
            heads=4, num_layers=1, pattern_heads=1, pattern_layers=1,
            pattern_dropout=pattern_dropout, dropout=dropout,
            use_relation_token=True, use_observed_relations=True,
        ))
        # The NC prediction head is replaced by the shared LP head; every active
        # encoder, relation adapter, anonymous GRU and original closure remains.
        self.node_model.backbone.head = nn.Identity()
        self.z_norm = nn.LayerNorm(hidden_dim) if z_layer_norm else nn.Identity()
        self.pair_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.U = nn.Parameter(torch.empty(8, hidden_dim, rank))
        self.V = nn.Parameter(torch.empty(8, hidden_dim, rank))
        nn.init.normal_(self.U, std=1.0 / math.sqrt(hidden_dim))
        nn.init.normal_(self.V, std=1.0 / math.sqrt(hidden_dim))
        # Fixed stabilization of each projected occurrence. Both sides carry
        # this factor, so F is scaled by 1/(8*r) relative to raw UV contraction.
        self.witness_projection_scale = 1.0 / math.sqrt(8 * rank)
        self.joint_scale = nn.Parameter(torch.tensor(float(joint_scale_init)))

    def encode(self, x_global: Tensor, paths: Tensor, relations: Tensor,
               occurrence_valid: Tensor | None = None) -> dict[str, Tensor]:
        """Encode roots; return z and sorted, compact padded witness tables.

        ``paths`` is [B,K,9]; ``relations`` is the actual [B,K,93] descriptor.
        Witness tables have M=K*8 slots, with invalid slots padded by N. Only
        t=1..8 contribute, and occurrence values are summed across walks / K.
        ``occurrence_valid`` may be [B,K,9] or [B,K,8]. If omitted, repeated
        consecutive positions are recognized as isolated-node SRW padding.
        """
        output = self.node_model(x_global, paths, relations)
        paths = paths.to(device=x_global.device, dtype=torch.long)
        contextual = output["contextual"][..., 1:, :]
        if occurrence_valid is None:
            occurrence_valid = paths[..., 1:] != paths[..., :-1]
        else:
            occurrence_valid = occurrence_valid.to(device=paths.device, dtype=torch.bool)
            if occurrence_valid.shape == paths.shape:
                occurrence_valid = occurrence_valid[..., 1:]
            if occurrence_valid.shape != paths[..., 1:].shape:
                raise ValueError("occurrence_valid must match [B,K,9] or [B,K,8]")
        batch, walks, length, _ = contextual.shape
        # Each depth has its own U_t,V_t before witness aggregation. This is a
        # low-rank factorization of the full depth-by-depth bilinear kernel.
        left_occ = torch.einsum("bktd,tdr->bktr", contextual, self.U)
        right_occ = torch.einsum("bktd,tdr->bktr", contextual, self.V)
        scale = self.witness_projection_scale / walks
        weights = occurrence_valid[..., None].to(contextual.dtype) * scale
        left_occ = (left_occ * weights).reshape(batch, walks * length, self.rank)
        right_occ = (right_occ * weights).reshape_as(left_occ)
        sentinel = x_global.shape[0]
        witness_ids = paths[..., 1:].masked_fill(~occurrence_valid, sentinel)
        sorted_ids, order = witness_ids.reshape(batch, -1).sort(dim=1)
        is_new = torch.ones_like(sorted_ids, dtype=torch.bool)
        is_new[:, 1:] = sorted_ids[:, 1:] != sorted_ids[:, :-1]
        groups = is_new.cumsum(dim=1) - 1
        ids = torch.full_like(sorted_ids, sentinel).scatter(1, groups, sorted_ids)
        gather_index = order[..., None].expand(-1, -1, self.rank)
        scatter_index = groups[..., None].expand_as(gather_index)
        left = left_occ.new_zeros(left_occ.shape).scatter_add(
            1, scatter_index, left_occ.gather(1, gather_index))
        right = right_occ.new_zeros(right_occ.shape).scatter_add(
            1, scatter_index, right_occ.gather(1, gather_index))
        return {"z": self.z_norm(output["z"]), "ids": ids,
                "left": left, "right": right, "valid": ids != sentinel}

    def joint_score(self, encoded: Mapping[str, Tensor], src_idx: Tensor,
                    dst_idx: Tensor) -> Tensor:
        """Symmetric contraction over matching IDs, without K²L² tensors."""
        device = encoded["z"].device
        src_idx = torch.as_tensor(src_idx, device=device, dtype=torch.long).reshape(-1)
        dst_idx = torch.as_tensor(dst_idx, device=device, dtype=torch.long).reshape(-1)
        if src_idx.shape != dst_idx.shape:
            raise ValueError("src_idx and dst_idx must have equal shape")
        outputs = []
        width = encoded["ids"].shape[1]
        for start in range(0, len(src_idx), self.pair_chunk_size):
            src, dst = src_idx[start:start + self.pair_chunk_size], dst_idx[start:start + self.pair_chunk_size]
            left_ids = encoded["ids"][src].contiguous()
            right_ids = encoded["ids"][dst].contiguous()
            positions = torch.searchsorted(right_ids, left_ids).clamp_max(width - 1)
            valid = (encoded["valid"][src]
                     & encoded["valid"][dst].gather(1, positions)
                     & (right_ids.gather(1, positions) == left_ids))
            gather_index = positions[..., None].expand(-1, -1, self.rank)
            a_u, b_u = encoded["left"][src], encoded["right"][src]
            a_v = encoded["left"][dst].gather(1, gather_index)
            b_v = encoded["right"][dst].gather(1, gather_index)
            values = 0.5 * ((a_u * b_v).sum(-1) + (b_u * a_v).sum(-1))
            outputs.append((values * valid.to(values.dtype)).sum(-1))
        return torch.cat(outputs) if outputs else encoded["z"].new_empty(0)

    def score(self, encoded: Mapping[str, Tensor], src_idx: Tensor,
              dst_idx: Tensor, use_joint: bool = True) -> Tensor:
        device = encoded["z"].device
        src_idx = torch.as_tensor(src_idx, device=device, dtype=torch.long).reshape(-1)
        dst_idx = torch.as_tensor(dst_idx, device=device, dtype=torch.long).reshape(-1)
        if src_idx.shape != dst_idx.shape:
            raise ValueError("src_idx and dst_idx must have equal shape")
        base = self.pair_head(encoded["z"][src_idx] * encoded["z"][dst_idx]).squeeze(-1)
        return base + self.joint_scale * self.joint_score(encoded, src_idx, dst_idx) if use_joint else base

    def forward(self, x_global: Tensor, paths: Tensor, relations: Tensor,
                src_idx: Tensor, dst_idx: Tensor, *, use_joint: bool = True,
                occurrence_valid: Tensor | None = None) -> Tensor:
        return self.score(self.encode(x_global, paths, relations, occurrence_valid),
                          src_idx, dst_idx, use_joint)

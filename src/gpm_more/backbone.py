"""Active node-task GPM operators, adapted from the upstream MIT source.

Source: Zehong-Wang/GPM, revision 2ed3ebf7df8d8401520e6ec3bdbd1c2b291b5784.
Upstream copyright (c) 2025 Zehong Wang; see third_party/GPM-LICENSE.
Only the active float-feature Transformer/anonymous-GRU node path is included.
"""
from __future__ import annotations

import copy
import torch
from torch import Tensor, nn


class PatternEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_dim = config.hidden_dim
        self.multiscale = tuple(config.multiscale)
        self.pe_weight = config.pe_weight
        self.pre_projection = nn.Linear(config.input_dim, config.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.pattern_heads,
            dim_feedforward=config.hidden_dim * 4, batch_first=True,
            dropout=config.pattern_dropout,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.pattern_layers)
        self.projection = nn.Identity()
        self.pe_rnn = nn.GRU(input_size=9, hidden_size=config.hidden_dim,
                             num_layers=1, batch_first=True)
        self.pe_projection = nn.Linear(config.hidden_dim, config.hidden_dim)

    def forward(self, paths: Tensor, features: Tensor) -> tuple[Tensor, Tensor]:
        # Preserve upstream K-major flattening and all-position mean pooling.
        patterns = paths.permute(1, 0, 2).contiguous()
        count, batch, length = patterns.shape
        gathered = features[patterns.reshape(-1)].view(count * batch, length, -1)
        gathered = self.pre_projection(gathered)
        mask = None
        if len(self.multiscale) > 1:
            visible = torch.zeros(count, batch, length, device=paths.device, dtype=torch.bool)
            for index, scale in enumerate(self.multiscale):
                start = int(index * count / len(self.multiscale))
                visible[start:, :, :scale + 1] = True
            mask = ~visible.view(count * batch, length)
        contextual_flat = self.encoder(gathered, src_key_padding_mask=mask)
        # Upstream averages query positions even when some are masked as keys.
        # Changing to a masked mean would be a different architecture.
        tokens = contextual_flat.mean(dim=1).view(count, batch, self.hidden_dim)
        identity = (patterns.unsqueeze(-1) == patterns.unsqueeze(-2)).reshape(-1, 9, 9).float()
        identity = identity.to(dtype=gathered.dtype)
        pe, _ = self.pe_rnn(identity)
        pe = self.pe_projection(pe.mean(dim=1)).view(count, batch, self.hidden_dim)
        tokens = tokens + pe * self.pe_weight
        contextual = contextual_flat.view(count, batch, 9, self.hidden_dim).permute(1, 0, 2, 3)
        return tokens, contextual


class GPMBackbone(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.pattern_encoder = PatternEncoder(config)
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim, nhead=config.heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout, norm_first=config.norm_first,
        )
        self.encoder = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.hidden_dim)
        self.use_attn_fusion = config.use_attn_fusion
        if self.use_attn_fusion:
            self.attn_layer = nn.Linear(config.hidden_dim, 1)
        self.head = nn.Linear(config.hidden_dim, config.num_outputs)

    def transformer_encode(self, tokens: Tensor) -> Tensor:
        for layer in self.encoder:
            tokens = tokens + layer(self.norm(tokens))
        return tokens

    def readout(self, tokens: Tensor) -> Tensor:
        if self.use_attn_fusion:
            weight = self.attn_layer(tokens).squeeze(-1).softmax(dim=0)
            return (weight.unsqueeze(-1) * tokens).sum(dim=0)
        return tokens.mean(dim=0)

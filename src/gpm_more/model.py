"""Standalone active raw_sum graph pattern model and relation-input feedback."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Mapping

import torch
from torch import Tensor, nn

from .backbone import GPMBackbone
from .feedback import scale_feedback
from .operators import closure_phase_pool, fixed_frequencies

NUM_RELATIONS = 93
NUM_ACTIONS = 2


@dataclass(frozen=True)
class ModelConfig:
    input_dim: int
    num_outputs: int
    hidden_dim: int = 256
    heads: int = 4
    num_layers: int = 1
    pattern_heads: int = 1
    pattern_layers: int = 1
    pattern_dropout: float = 0.1
    dropout: float = 0.0
    norm_first: bool = False
    multiscale: tuple[int, ...] = (2, 4, 6, 8)
    pe_weight: float = 1.0
    use_attn_fusion: bool = False
    use_relation_token: bool = True
    use_observed_relations: bool = True

    def __post_init__(self):
        if min(self.input_dim, self.num_outputs, self.hidden_dim,
               self.heads, self.num_layers, self.pattern_heads, self.pattern_layers) < 1:
            raise ValueError("dimensions, heads, and layer counts must be positive")
        if self.hidden_dim % self.heads or self.hidden_dim % self.pattern_heads:
            raise ValueError("hidden_dim must be divisible by both head counts")
        if not self.multiscale or any(not 0 <= size <= 8 for size in self.multiscale):
            raise ValueError("multiscale entries must be in [0,8]")
        if tuple(sorted(self.multiscale)) != tuple(self.multiscale):
            raise ValueError("multiscale entries must be sorted")

    def to_dict(self):
        return asdict(self)


class RelationalPatternModel(nn.Module):
    """GPM active node backbone with the existing sum-phase relation Token.

    Default dimensions and active operations match the historical raw_sum256
    experiments. Disabled VQ, inactive source template layers, frozen legacy
    phase modules, and unused pretraining heads are intentionally omitted.
    Historical state dictionaries can be loaded with ``load_historical_state``.
    Set BOTH relation flags false for the active feature-only GPM path; turning
    off only the relation Token is a matched closure ablation, not published GPM.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.backbone = GPMBackbone(config)
        if config.use_observed_relations:
            self.relation_adapter = nn.Sequential(
                nn.Linear(NUM_RELATIONS, config.hidden_dim), nn.GELU(),
                nn.Linear(config.hidden_dim, config.hidden_dim),
            )
            self.action_embedding = nn.Embedding(NUM_ACTIONS, config.hidden_dim)
            nn.init.normal_(self.action_embedding.weight, mean=0.0, std=0.02)
        self.register_buffer("closure_frequencies", fixed_frequencies(config.hidden_dim), persistent=False)

    def forward(self, x: Tensor, paths: Tensor, relations: Tensor | None,
                action_ids: Tensor | None = None, feedback_alpha: float = 1.0) -> dict[str, Tensor]:
        if x.ndim != 2 or x.shape[-1] != self.config.input_dim:
            raise ValueError("x must have shape [N,input_dim]")
        if paths.ndim != 3 or paths.shape[-1] != 9 or paths.shape[1] < 1:
            raise ValueError("paths must contain nonempty [B,K,9] observed walks")
        if paths.dtype not in (torch.int32, torch.int64):
            raise ValueError("paths must contain integer node IDs")
        paths = paths.to(device=x.device, dtype=torch.long)
        if bool((paths < 0).any()) or bool((paths >= len(x)).any()):
            raise ValueError("paths contain invalid node indices")
        if action_ids is not None:
            if action_ids.shape != paths.shape[:2]:
                raise ValueError("action_ids must match [B,K]")
            action_ids = action_ids.to(device=x.device, dtype=torch.long)
            if bool((action_ids < 0).any()) or bool((action_ids >= NUM_ACTIONS).any()):
                raise ValueError("action_ids must contain only 0 or 1")
        tokens, contextual = self.backbone.pattern_encoder(paths, x)
        if self.config.use_observed_relations:
            if relations is None or relations.shape != (*paths.shape[:2], NUM_RELATIONS):
                raise ValueError("relations must have matching [B,K,93] shape")
            relation_tokens = self.relation_adapter(relations.to(x)).permute(1, 0, 2)
            actions = (self.action_embedding.weight[0].view(1, 1, -1)
                       if action_ids is None else self.action_embedding(action_ids).permute(1, 0, 2))
            tokens = tokens + relation_tokens + actions
        original_tokens = tokens
        if self.config.use_relation_token:
            closure = closure_phase_pool(paths, scale_feedback(contextual, feedback_alpha),
                                         self.closure_frequencies)
            tokens = torch.cat((tokens, closure[None]), dim=0)
        else:
            closure = contextual.new_zeros(paths.shape[0], self.config.hidden_dim)
        encoded = self.backbone.transformer_encode(tokens)
        z = self.backbone.readout(encoded)
        return {"prediction": self.backbone.head(z), "z": z,
                "closure_token": closure, "pattern_tokens": original_tokens,
                "contextual": contextual}

    def load_historical_state(self, state: Mapping[str, Tensor]) -> list[str]:
        """Load all active weights from a historical raw_sum state dictionary.

        Return explicitly ignored inactive keys. Missing active weights and
        shape differences are errors. This transfers weights, not Adam/RNG
        state or a historical training trajectory.
        """
        own = self.state_dict()
        missing = set(own) - set(state)
        if missing:
            raise ValueError(f"Historical state lacks active weights: {sorted(missing)}")
        ignored = sorted(set(state) - set(own))
        allowed = ("backbone.vq.", "backbone.encoder_layer.", "phase_encoder.",
                   "phase_projection.", "structural_head.", "value_head.")
        unexpected = [name for name in ignored if name != "phase_gate" and not name.startswith(allowed)]
        if unexpected:
            raise ValueError(f"Unrecognized historical keys: {unexpected}")
        self.load_state_dict({name: state[name] for name in own}, strict=True)
        return ignored

    load_legacy_state_dict = load_historical_state

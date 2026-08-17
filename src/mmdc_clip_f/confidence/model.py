"""The encoder-independent MV-ACN confidence head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class MVACNConfig:
    hidden_dim: int
    attention_heads: int = 4
    mlp_dims: tuple[int, ...] = (128, 64)
    dropout: float = 0.2
    drop_cls_tokens: bool = True

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.attention_heads <= 0:
            raise ValueError("attention_heads must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not self.mlp_dims or any(dimension <= 0 for dimension in self.mlp_dims):
            raise ValueError("mlp_dims must contain positive dimensions")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class MVACNHead(nn.Module):
    """Attention confidence head over four precomputed vision hidden states.

    ``forward`` returns logits for use with ``BCEWithLogitsLoss``. Use
    :meth:`predict_confidence` when a calibrated ``[0, 1]`` score is required.
    The frozen Stage-1 encoder is intentionally not owned by this module.
    """

    def __init__(self, config: MVACNConfig):
        super().__init__()
        self.config = config
        self.confidence_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        nn.init.trunc_normal_(self.confidence_token, std=0.02)
        self.attention = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.attention_heads,
            batch_first=True,
        )
        self.normalization = nn.LayerNorm(config.hidden_dim)
        layers: list[nn.Module] = []
        input_dim = config.hidden_dim
        for output_dim in config.mlp_dims:
            layers.extend(
                [
                    nn.Linear(input_dim, output_dim),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                ]
            )
            input_dim = output_dim
        layers.append(nn.Linear(input_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _validate_views(self, views: Sequence[Tensor]) -> None:
        if len(views) != 4:
            raise ValueError(f"MV-ACN requires exactly four views, received {len(views)}")
        reference = views[0]
        if reference.ndim != 3:
            raise ValueError("each view must have shape [batch, tokens, hidden_dim]")
        batch_size = reference.shape[0]
        for index, view in enumerate(views):
            if view.ndim != 3:
                raise ValueError(f"view {index} is not a rank-3 tensor")
            if view.shape[0] != batch_size:
                raise ValueError("all views must have the same batch size")
            if view.shape[2] != self.config.hidden_dim:
                raise ValueError(
                    f"view {index} hidden dimension {view.shape[2]} does not match "
                    f"configured {self.config.hidden_dim}"
                )
            if self.config.drop_cls_tokens and view.shape[1] < 2:
                raise ValueError("cannot drop CLS from a view with fewer than two tokens")

    def forward(self, *views: Tensor) -> Tensor:
        self._validate_views(views)
        prepared = tuple(view[:, 1:, :] for view in views) if self.config.drop_cls_tokens else views
        hidden_states = torch.cat(prepared, dim=1)
        confidence_token = self.confidence_token.expand(hidden_states.shape[0], -1, -1)
        hidden_states = torch.cat((confidence_token, hidden_states), dim=1)
        attended, _ = self.attention(
            hidden_states, hidden_states, hidden_states, need_weights=False
        )
        pooled = self.normalization(attended[:, 0, :])
        return self.mlp(pooled).squeeze(-1)

    def predict_confidence(self, *views: Tensor) -> Tensor:
        return torch.sigmoid(self(*views))


def export_head_state_dict(model: MVACNHead) -> dict[str, Tensor]:
    """Return CPU, contiguous, tensor-only weights accepted by safetensors."""

    if not isinstance(model, MVACNHead):
        raise TypeError("model must be an MVACNHead")
    return {
        name: value.detach().cpu().contiguous().clone()
        for name, value in model.state_dict().items()
    }

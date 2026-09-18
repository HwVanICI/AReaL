# SPDX-License-Identifier: Apache-2.0

"""Policy-gradient loss aggregation and distributed normalizer contracts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import torch

LossAggregationMode = Literal[
    "token-mean", "seq-mean", "traj-mean", "prompt-mean", "constant"
]
_LOSS_AGGREGATIONS = (
    "token-mean",
    "seq-mean",
    "traj-mean",
    "prompt-mean",
    "constant",
)
# Reductions carrying their denominator as a precomputed per-token column.
_WEIGHTED_AGGREGATIONS = ("traj-mean", "prompt-mean")


@dataclass(frozen=True, slots=True)
class PolicyGradientReduction:
    """Policy-gradient microbatch mean and its matching engine weight.

    The training engine combines microbatches as
    ``sum(local_mean * local_weight) / sum(local_weight)``.

    ``traj-mean`` and ``prompt-mean`` instead take a precomputed per-token
    ``unit_weights`` column holding ``1 / (tokens in this token's unit)`` -- the
    unit being one trajectory, or one whole rollout group. PPO optimizer
    minibatches keep a unit's rows atomic. Within one optimizer step the unit
    may still be split across forward microbatches and data-parallel ranks:
    summing the weights recovers the unit count, which is exactly the
    denominator the engine divides by.
    """

    mode: LossAggregationMode = "token-mean"
    divisor: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in _LOSS_AGGREGATIONS:
            raise ValueError(
                f"loss_aggregation must be one of {_LOSS_AGGREGATIONS}, "
                f"got {self.mode!r}."
            )
        if self.mode == "constant":
            if (
                self.divisor is None
                or not math.isfinite(self.divisor)
                or self.divisor <= 0
            ):
                raise ValueError(
                    "divisor must be a positive finite value for "
                    "loss_aggregation='constant'."
                )
        elif self.divisor is not None:
            raise ValueError("divisor is only valid for loss_aggregation='constant'.")

    def normalizer_fn(self, data: dict[str, Any]) -> torch.Tensor:
        """Return this reduction's active local denominator."""
        loss_mask = data["loss_mask"].bool()
        if self.mode == "token-mean":
            return loss_mask.count_nonzero()
        if self.mode in _WEIGHTED_AGGREGATIONS:
            return self._unit_weight_sum(data.get("unit_weights"))

        self._require_sequence_boundaries(loss_mask, data.get("cu_seqlens"))
        sequence_denominators = self._sequence_sums(
            loss_mask.to(torch.float32), data.get("cu_seqlens")
        )
        ids, n_units = self._unit_ids(sequence_denominators.numel(), loss_mask.device)
        unit_denominators = torch.zeros(
            n_units, dtype=torch.float32, device=loss_mask.device
        )
        unit_denominators.scatter_add_(0, ids, sequence_denominators)
        return unit_denominators.count_nonzero().to(torch.float32)

    def aggregate(
        self,
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        *,
        denominator_mask: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        unit_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Aggregate a token-shaped policy-gradient loss."""
        numerator_mask, denominator_mask = self._resolve_masks(
            loss, loss_mask, denominator_mask
        )
        if self.mode == "token-mean":
            # Preserve the pre-feature token-mean dtype and reduction path.
            numerator = torch.where(numerator_mask, loss, 0).sum()
            return numerator / denominator_mask.count_nonzero().clamp_min(1)

        if self.mode in _WEIGHTED_AGGREGATIONS:
            weights = self._require_unit_weights(loss, unit_weights)
            numerator = (self._masked_loss(loss, numerator_mask) * weights).sum()
            return numerator / self._unit_weight_sum(weights).clamp_min(
                torch.finfo(torch.float32).tiny
            )

        self._require_sequence_boundaries(loss_mask, cu_seqlens)
        if self.mode == "constant":
            divisor = self._require_divisor()
            numerator = self._masked_loss(loss, numerator_mask).sum()
            active_sequences = self._sequence_sums(
                denominator_mask.to(torch.float32), cu_seqlens
            ).count_nonzero()
            return numerator / (active_sequences.clamp_min(1) * divisor)

        return self._aggregate_units(
            loss, numerator_mask, denominator_mask, cu_seqlens=cu_seqlens
        )

    @staticmethod
    def _unit_weight_sum(unit_weights: torch.Tensor | None) -> torch.Tensor:
        if unit_weights is None:
            raise ValueError(
                "unit_weights are required for loss_aggregation in "
                f"{_WEIGHTED_AGGREGATIONS}."
            )
        return unit_weights.to(torch.float32).sum()

    @staticmethod
    def _require_unit_weights(
        loss: torch.Tensor, unit_weights: torch.Tensor | None
    ) -> torch.Tensor:
        if unit_weights is None:
            raise ValueError(
                "unit_weights are required for loss_aggregation in "
                f"{_WEIGHTED_AGGREGATIONS}."
            )
        if unit_weights.shape != loss.shape:
            raise ValueError(
                f"unit_weights shape {tuple(unit_weights.shape)} must match "
                f"loss shape {tuple(loss.shape)}."
            )
        return unit_weights.to(torch.float32)

    def _require_divisor(self) -> float:
        if self.divisor is None:
            raise ValueError(
                "a positive divisor is required for loss_aggregation='constant'."
            )
        return self.divisor

    def _require_sequence_boundaries(
        self, loss_mask: torch.Tensor, cu_seqlens: torch.Tensor | None
    ) -> None:
        if cu_seqlens is None and loss_mask.ndim == 1:
            raise ValueError(
                f"loss_aggregation='{self.mode}' requires cu_seqlens for packed "
                "inputs; tree-packed training currently supports only token-mean."
            )

    @staticmethod
    def _masked_loss(loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask, loss, 0).to(torch.float32)

    @staticmethod
    def _resolve_masks(
        loss: torch.Tensor,
        loss_mask: torch.Tensor,
        denominator_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if loss.shape != loss_mask.shape:
            raise ValueError(
                f"loss_mask shape {tuple(loss_mask.shape)} must match "
                f"loss shape {tuple(loss.shape)}."
            )
        if denominator_mask is not None and loss.shape != denominator_mask.shape:
            raise ValueError(
                f"denom_mask shape {tuple(denominator_mask.shape)} must match "
                f"loss shape {tuple(loss.shape)}."
            )
        numerator_mask = loss_mask.bool()
        return numerator_mask, (
            numerator_mask if denominator_mask is None else denominator_mask.bool()
        )

    @staticmethod
    def _sequence_sums(
        values: torch.Tensor, cu_seqlens: torch.Tensor | None
    ) -> torch.Tensor:
        """Sum token values per sequence for padded or packed inputs."""
        if cu_seqlens is None:
            if values.ndim != 2:
                raise ValueError(
                    "padded policy-gradient inputs must be 2D, "
                    f"got shape {tuple(values.shape)}."
                )
            return values.sum(dim=-1)

        if values.ndim != 1:
            raise ValueError(
                "packed policy-gradient inputs must be 1D, "
                f"got shape {tuple(values.shape)}."
            )
        if cu_seqlens.ndim != 1:
            raise ValueError(
                f"cu_seqlens must be 1D, got shape {tuple(cu_seqlens.shape)}."
            )

        n_sequences = cu_seqlens.numel() - 1
        sequence_lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).to(
            device=values.device, dtype=torch.long
        )
        sequence_ids = torch.arange(
            n_sequences, device=values.device
        ).repeat_interleave(sequence_lengths, output_size=values.numel())
        if sequence_ids.numel() != values.numel():
            raise ValueError(
                "cu_seqlens does not describe the packed loss: "
                f"expected {sequence_ids.numel()} tokens, got {values.numel()}."
            )
        result = torch.zeros(n_sequences, dtype=values.dtype, device=values.device)
        return result.scatter_add_(0, sequence_ids, values)

    @staticmethod
    def _unit_ids(n_sequences: int, device: torch.device) -> tuple[torch.Tensor, int]:
        return torch.arange(n_sequences, device=device), n_sequences

    @staticmethod
    def _reduce_unit_means(
        numerator: torch.Tensor,
        denominator: torch.Tensor,
    ) -> torch.Tensor:
        active = denominator > 0
        unit_means = torch.where(
            active,
            numerator / denominator.clamp_min(1),
            torch.zeros_like(numerator),
        )
        return unit_means.sum() / active.count_nonzero().clamp_min(1)

    @classmethod
    def _aggregate_units(
        cls,
        loss: torch.Tensor,
        numerator_mask: torch.Tensor,
        denominator_mask: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        masked_loss = cls._masked_loss(loss, numerator_mask)
        sequence_numerators = cls._sequence_sums(masked_loss, cu_seqlens)
        sequence_denominators = cls._sequence_sums(
            denominator_mask.to(torch.float32), cu_seqlens
        )
        ids, n_units = cls._unit_ids(sequence_numerators.numel(), loss.device)
        unit_numerators = torch.zeros(n_units, dtype=torch.float32, device=loss.device)
        unit_denominators = torch.zeros(
            n_units, dtype=torch.float32, device=loss.device
        )
        unit_numerators.scatter_add_(0, ids, sequence_numerators)
        unit_denominators.scatter_add_(0, ids, sequence_denominators)
        return cls._reduce_unit_means(unit_numerators, unit_denominators)

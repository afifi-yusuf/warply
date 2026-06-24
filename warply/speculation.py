from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from warply.exceptions import ValidationError

SpeculationMode = Literal["none", "engine_native", "mtp", "eagle", "dflash", "draft_model"]
SUPPORTED_SPECULATION_MODES = frozenset(
    {"none", "engine_native", "mtp", "eagle", "dflash", "draft_model"}
)


@dataclass(frozen=True)
class Speculation:
    """Speculative decoding intent for a deployment."""

    mode: SpeculationMode = "none"
    draft_model: str | None = None
    speculator_model: str | None = None
    max_draft_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in SUPPORTED_SPECULATION_MODES:
            options = ", ".join(sorted(SUPPORTED_SPECULATION_MODES))
            raise ValidationError(
                f"Unsupported speculation mode {self.mode!r}; expected one of: {options}."
            )
        if self.mode == "none" and (
            self.draft_model is not None
            or self.speculator_model is not None
            or self.max_draft_tokens is not None
        ):
            raise ValidationError(
                "mode='none' cannot set draft_model, speculator_model, or max_draft_tokens."
            )
        if self.mode == "draft_model" and not self.draft_model:
            raise ValidationError("mode='draft_model' requires draft_model.")
        if self.mode in {"eagle", "dflash"} and not self.speculator_model:
            raise ValidationError(f"mode={self.mode!r} requires speculator_model.")
        if self.max_draft_tokens is not None and self.max_draft_tokens < 1:
            raise ValidationError("max_draft_tokens must be >= 1.")

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "mode": self.mode,
            "draft_model": self.draft_model,
            "speculator_model": self.speculator_model,
            "max_draft_tokens": self.max_draft_tokens,
        }

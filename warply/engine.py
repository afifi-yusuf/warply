from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from warply._constants import SUPPORTED_BACKENDS, SUPPORTED_CLOUDS, SUPPORTED_KV_TRANSFERS
from warply.exceptions import NotReadyError, ValidationError
from warply.pool import Pool
from warply.types import DeploymentStatus, EngineState, PoolStatus


def _validate_choice(field: str, value: str, allowed: frozenset[str]) -> None:
    if value not in allowed:
        options = ", ".join(sorted(allowed))
        raise ValidationError(f"Unsupported {field} {value!r}; expected one of: {options}.")


@dataclass
class DisaggEngine:
    """Declarative spec + lifecycle control for a disaggregated deployment."""

    model: str
    prefill: Pool
    decode: Pool
    backend: str = "sglang"
    kv_transfer: str = "nixl"
    cloud: str = "local"
    _state: EngineState = field(default=EngineState.PENDING, init=False, repr=False)
    _prefill_replicas: int | None = field(default=None, init=False, repr=False)
    _decode_replicas: int | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValidationError("model must be a non-empty Hugging Face repo id or local path.")

        _validate_choice("backend", self.backend, SUPPORTED_BACKENDS)
        _validate_choice("kv_transfer", self.kv_transfer, SUPPORTED_KV_TRANSFERS)
        _validate_choice("cloud", self.cloud, SUPPORTED_CLOUDS)

        if self._prefill_replicas is None:
            self._prefill_replicas = self.prefill.replicas
        if self._decode_replicas is None:
            self._decode_replicas = self.decode.replicas

    def up(self) -> DisaggEngine:
        """Provision, deploy, and wire routing. Blocks until ready."""
        raise NotImplementedError(
            "DisaggEngine.up() is not implemented yet. "
            "See CONTRIBUTING.md for the build sequence."
        )

    def scale(self, *, prefill: int | None = None, decode: int | None = None) -> DisaggEngine:
        """Independently resize either pool."""
        if prefill is None and decode is None:
            raise ValidationError("scale() requires at least one of prefill= or decode=.")

        if prefill is not None and prefill < 1:
            raise ValidationError(f"prefill replicas must be >= 1, got {prefill}.")
        if decode is not None and decode < 1:
            raise ValidationError(f"decode replicas must be >= 1, got {decode}.")

        if self._state not in {EngineState.READY, EngineState.SCALING}:
            raise NotReadyError("scale() requires a running deployment. Call up() first.")

        raise NotImplementedError(
            "DisaggEngine.scale() is not implemented yet. "
            "See CONTRIBUTING.md for the build sequence."
        )

    def status(self) -> DeploymentStatus:
        """Return structured deployment state."""
        return DeploymentStatus(
            state=self._state,
            model=self.model,
            backend=self.backend,
            kv_transfer=self.kv_transfer,
            cloud=self.cloud,
            prefill=PoolStatus(
                gpus=self.prefill.gpus,
                replicas=self._prefill_replicas or self.prefill.replicas,
            ),
            decode=PoolStatus(
                gpus=self.decode.gpus,
                replicas=self._decode_replicas or self.decode.replicas,
            ),
            endpoint=None,
        )

    def down(self) -> None:
        """Tear down the deployment."""
        if self._state == EngineState.PENDING:
            self._state = EngineState.STOPPED
            return

        raise NotImplementedError(
            "DisaggEngine.down() is not implemented yet. "
            "See CONTRIBUTING.md for the build sequence."
        )

    def client(self) -> Any:
        """Return an OpenAI-compatible client bound to this deployment."""
        if self._state != EngineState.READY:
            raise NotReadyError("client() requires a ready deployment. Call up() first.")

        raise NotImplementedError(
            "DisaggEngine.client() is not implemented yet. "
            "See CONTRIBUTING.md for the build sequence."
        )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Convenience wrapper around the OpenAI-compatible client."""
        if not prompt.strip():
            raise ValidationError("prompt must be non-empty.")

        client = self.client()
        response = client.chat.completions.create(
            model="warply",
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response.choices[0].message.content

    def export_yaml(self) -> str:
        """Emit a raw deployment manifest for power users."""
        raise NotImplementedError(
            "DisaggEngine.export_yaml() is not implemented yet. "
            "See CONTRIBUTING.md for the build sequence."
        )

    def __enter__(self) -> DisaggEngine:
        return self.up()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.down()

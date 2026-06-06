from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

from warply._constants import SUPPORTED_BACKENDS, SUPPORTED_CLOUDS, SUPPORTED_KV_TRANSFERS
from warply.compiler import compile
from warply.compiler.plan import DeploymentPlan
from warply.exceptions import NotReadyError, ValidationError
from warply.pool import Pool
from warply.runtime.client import completion_content
from warply.runtime.factory import create_runtime
from warply.runtime.lifecycle import state_after_down
from warply.runtime.yaml import dump_yaml
from warply.types import DeploymentStatus, EngineState, PoolStatus

if TYPE_CHECKING:
    from warply.runtime.lifecycle import Runtime

PoolRole = Literal["prefill", "decode"]


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
    _runtime: Runtime | None = field(default=None, init=False, repr=False)

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
        if self._state in {EngineState.PROVISIONING, EngineState.READY, EngineState.SCALING}:
            raise NotReadyError(f"up() cannot start while deployment is {self._state.value}.")

        self._state = EngineState.PROVISIONING
        self._runtime = create_runtime(self.plan())
        try:
            self._runtime.up()
        except Exception:
            self._runtime.down()
            self._runtime = None
            self._state = EngineState.STOPPED
            raise

        self._state = EngineState.READY
        return self

    def scale(self, *, prefill: int | None = None, decode: int | None = None) -> DisaggEngine:
        """Independently resize either pool."""
        if self.cloud != "local":
            raise NotImplementedError(
                "cloud scale() is not implemented yet. Tear down and relaunch with a new spec."
            )

        if prefill is None and decode is None:
            raise ValidationError("scale() requires at least one of prefill= or decode=.")

        if prefill is not None and prefill < 1:
            raise ValidationError(f"prefill replicas must be >= 1, got {prefill}.")
        if decode is not None and decode < 1:
            raise ValidationError(f"decode replicas must be >= 1, got {decode}.")

        if self._state not in {EngineState.READY, EngineState.SCALING}:
            raise NotReadyError("scale() requires a running deployment. Call up() first.")

        if self._runtime is None:
            raise NotReadyError("scale() requires an active runtime. Call up() first.")

        old_prefill_replicas = self._prefill_replicas
        old_decode_replicas = self._decode_replicas
        if prefill is not None:
            self._prefill_replicas = prefill
        if decode is not None:
            self._decode_replicas = decode

        new_plan = self.plan()
        self._state = EngineState.SCALING
        try:
            self._runtime.scale(plan=new_plan, prefill=prefill, decode=decode)
        except Exception:
            self._prefill_replicas = old_prefill_replicas
            self._decode_replicas = old_decode_replicas
            self._state = EngineState.READY
            raise

        self._state = EngineState.READY
        return self

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
                healthy_replicas=self._runtime.healthy_prefill() if self._runtime else 0,
            ),
            decode=PoolStatus(
                gpus=self.decode.gpus,
                replicas=self._decode_replicas or self.decode.replicas,
                healthy_replicas=self._runtime.healthy_decode() if self._runtime else 0,
            ),
            endpoint=self._runtime.endpoint if self._runtime else None,
        )

    def down(self) -> None:
        """Tear down the deployment."""
        if self._runtime is not None:
            self._state = EngineState.STOPPING
            self._runtime.down()
            self._runtime = None

        next_state = state_after_down(self._state)
        if next_state is EngineState.STOPPED:
            self._state = next_state
            return

        raise NotImplementedError(
            "DisaggEngine.down() is not implemented yet. "
            "See CONTRIBUTING.md for the build sequence."
        )

    def client(self) -> Any:
        """Return an OpenAI-compatible client bound to this deployment."""
        if self._state != EngineState.READY:
            raise NotReadyError("client() requires a ready deployment. Call up() first.")

        if self._runtime is None:
            raise NotReadyError("client() requires an active runtime. Call up() first.")
        return self._runtime.client()

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
        return completion_content(response)

    def export_yaml(self) -> str:
        """Emit a raw deployment manifest for power users."""
        return dump_yaml(self.plan().to_dict())

    def plan(self) -> DeploymentPlan:
        """Return the compiled deployment plan for debugging and adapters."""
        return compile(self)

    def deployed_plan(self) -> DeploymentPlan | None:
        """Return the runtime-resolved plan after up(), if available."""
        if self._runtime is None:
            return None
        return self._runtime.plan

    def effective_replicas(self, role: PoolRole) -> int:
        """Return the current desired replica count for a pool role."""
        if role == "prefill":
            return self._prefill_replicas or self.prefill.replicas
        return self._decode_replicas or self.decode.replicas

    def __enter__(self) -> DisaggEngine:
        return self.up()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.down()

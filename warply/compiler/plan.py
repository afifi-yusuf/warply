from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from warply.accelerators import AcceleratorProfile

PoolRole = Literal["prefill", "decode"]


@dataclass(frozen=True)
class ProvisionRequest:
    """Provider-facing request for one disaggregated pool."""

    role: PoolRole
    cloud: str
    gpu_type: str
    gpus_per_replica: int
    replicas: int
    accelerator: AcceleratorProfile

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "cloud": self.cloud,
            "gpu_type": self.gpu_type,
            "gpus_per_replica": self.gpus_per_replica,
            "replicas": self.replicas,
            "accelerator": self.accelerator.to_dict(),
        }


@dataclass(frozen=True)
class PoolPlan:
    """Normalized pool plan shared by providers, engine adapters, and routers."""

    role: PoolRole
    gpus: str
    gpu_type: str
    gpus_per_replica: int
    replicas: int
    base_port: int
    accelerator: AcceleratorProfile
    provision: ProvisionRequest

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "gpus": self.gpus,
            "gpu_type": self.gpu_type,
            "gpus_per_replica": self.gpus_per_replica,
            "replicas": self.replicas,
            "base_port": self.base_port,
            "accelerator": self.accelerator.to_dict(),
            "provision": self.provision.to_dict(),
        }


@dataclass(frozen=True)
class RoutingConfig:
    """Router-facing config for a prefill/decode deployment."""

    mode: str
    router_port: int
    endpoint: str
    prefill_base_url: str
    decode_base_url: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "router_port": self.router_port,
            "endpoint": self.endpoint,
            "prefill_base_url": self.prefill_base_url,
            "decode_base_url": self.decode_base_url,
        }


@dataclass(frozen=True)
class DeploymentPlan:
    """Canonical compiler artifact for a Warply deployment."""

    model: str
    backend: str
    kv_transfer: str
    resolved_kv_transfer: str | None
    cloud: str
    prefill: PoolPlan
    decode: PoolPlan
    routing: RoutingConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "backend": self.backend,
            "kv_transfer": self.kv_transfer,
            "resolved_kv_transfer": self.resolved_kv_transfer,
            "cloud": self.cloud,
            "prefill": self.prefill.to_dict(),
            "decode": self.decode.to_dict(),
            "routing": self.routing.to_dict(),
        }

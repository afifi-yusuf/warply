from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class AcceleratorProfile:
    """Hardware family metadata derived from a pool GPU type."""

    gpu_type: str
    vendor: str
    runtime: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "gpu_type": self.gpu_type,
            "vendor": self.vendor,
            "runtime": self.runtime,
        }


_NVIDIA_GPUS = frozenset({"A100", "H100", "H200", "B200"})
_AMD_GPUS = frozenset({"MI250", "MI250X", "MI300", "MI300X", "MI325X", "MI350", "MI350X"})


def accelerator_profile(gpu_type: str) -> AcceleratorProfile:
    """Return vendor/runtime metadata for known GPU families."""
    normalized = gpu_type.upper()
    if normalized in _NVIDIA_GPUS:
        return AcceleratorProfile(gpu_type=gpu_type, vendor="nvidia", runtime="cuda")
    if normalized in _AMD_GPUS:
        return AcceleratorProfile(gpu_type=gpu_type, vendor="amd", runtime="rocm")
    return AcceleratorProfile(gpu_type=gpu_type, vendor="unknown", runtime="unknown")

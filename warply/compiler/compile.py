from __future__ import annotations

from typing import TYPE_CHECKING

from warply.accelerators import accelerator_profile
from warply.compiler.plan import DeploymentPlan, PoolPlan, PoolRole, ProvisionRequest, RoutingConfig
from warply.pool import Pool

if TYPE_CHECKING:
    from warply.engine import DisaggEngine


PREFILL_PORT = 31000
DECODE_PORT = 32000
ROUTER_PORT = 8000


def _parse_gpu_spec(gpus: str) -> tuple[int, str]:
    count, gpu_type = gpus.split("x", maxsplit=1)
    return int(count), gpu_type


def _pool_plan(
    *,
    role: PoolRole,
    pool: Pool,
    cloud: str,
    base_port: int,
    replicas: int,
) -> PoolPlan:
    gpu_count, gpu_type = _parse_gpu_spec(pool.gpus)
    accelerator = accelerator_profile(gpu_type)
    provision = ProvisionRequest(
        role=role,
        cloud=cloud,
        gpu_type=gpu_type,
        gpus_per_replica=gpu_count,
        replicas=replicas,
        accelerator=accelerator,
    )
    return PoolPlan(
        role=role,
        gpus=pool.gpus,
        gpu_type=gpu_type,
        gpus_per_replica=gpu_count,
        replicas=replicas,
        base_port=base_port,
        accelerator=accelerator,
        provision=provision,
    )


def _routing_config(cloud: str) -> RoutingConfig:
    if cloud == "local":
        endpoint = f"http://127.0.0.1:{ROUTER_PORT}"
        prefill_url = f"http://127.0.0.1:{PREFILL_PORT}"
        decode_url = f"http://127.0.0.1:{DECODE_PORT}"
    else:
        endpoint = f"warply://{cloud}/router"
        prefill_url = f"warply://{cloud}/prefill"
        decode_url = f"warply://{cloud}/decode"

    return RoutingConfig(
        mode="prefill_decode",
        router_port=ROUTER_PORT,
        endpoint=endpoint,
        prefill_base_url=prefill_url,
        decode_base_url=decode_url,
    )


def _resolve_kv_transfer(kv_transfer: str, prefill: PoolPlan, decode: PoolPlan) -> str | None:
    both_cuda = prefill.accelerator.runtime == decode.accelerator.runtime == "cuda"
    if kv_transfer == "nixl" and both_cuda:
        return kv_transfer
    if kv_transfer == "auto" and both_cuda:
        return "nixl"
    return None


def compile(engine: DisaggEngine) -> DeploymentPlan:
    """Compile a DisaggEngine spec into a deterministic deployment plan."""
    prefill = _pool_plan(
        role="prefill",
        pool=engine.prefill,
        cloud=engine.cloud,
        base_port=PREFILL_PORT,
        replicas=engine.effective_replicas("prefill"),
    )
    decode = _pool_plan(
        role="decode",
        pool=engine.decode,
        cloud=engine.cloud,
        base_port=DECODE_PORT,
        replicas=engine.effective_replicas("decode"),
    )
    return DeploymentPlan(
        model=engine.model,
        backend=engine.backend,
        kv_transfer=engine.kv_transfer,
        resolved_kv_transfer=_resolve_kv_transfer(engine.kv_transfer, prefill, decode),
        cloud=engine.cloud,
        prefill=prefill,
        decode=decode,
        routing=_routing_config(engine.cloud),
    )

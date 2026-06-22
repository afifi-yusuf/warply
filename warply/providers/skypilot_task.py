from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import TYPE_CHECKING

from warply.runtime.yaml import dump_yaml

if TYPE_CHECKING:
    from warply.compiler.plan import DeploymentPlan, PoolPlan

_CLOUD_ALIASES = {
    "lambda": "lambda",
    "coreweave": "coreweave",
}

_WORKER_SETUP = """\
pip install -U pip
pip install 'sglang[all]'
pip install 'nixl[cu12]' || pip install nixl
"""

_ROCM_WORKER_SETUP = """\
echo "AMD/ROCm SGLang launch is not enabled in Warply yet." >&2
exit 1
"""

_ROUTER_SETUP = """\
pip install -U pip
pip install 'sglang[all]'
"""

_WAIT_HTTP_SCRIPT = """\
wait_for_http() {
  local url="$1"
  local label="$2"
  local pid="${3:-}"
  local attempts="${4:-120}"
  local delay="${5:-5}"
  local i=1
  while [ "$i" -le "$attempts" ]; do
    if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
      echo "process for ${label} exited before becoming ready" >&2
      wait "$pid" || true
      return 1
    fi
    if python - "$url" <<'PY'
import sys
import urllib.request

try:
    with urllib.request.urlopen(sys.argv[1], timeout=5) as response:
        sys.exit(0 if 200 <= response.status < 300 else 1)
except Exception:
    sys.exit(1)
PY
    then
      echo "${label} ready at ${url}"
      return 0
    fi
    echo "waiting for ${label} (${i}/${attempts})..."
    sleep "$delay"
    i=$((i + 1))
  done
  echo "timed out waiting for ${label} at ${url}" >&2
  return 1
}
"""


@dataclass(frozen=True)
class ClusterPlacement:
    prefill_ranks: list[int]
    decode_ranks: list[int]

    @property
    def num_nodes(self) -> int:
        return len(self.prefill_ranks) + len(self.decode_ranks)


def cluster_name(*, session_id: str, role: str, index: int) -> str:
    return f"warply-{session_id}-{role}-{index}"


def router_cluster_name(*, session_id: str) -> str:
    return f"warply-{session_id}-router"


def disagg_cluster_name(*, session_id: str) -> str:
    return f"warply-{session_id}-disagg"


def accelerator_spec(gpu_type: str, count: int) -> str:
    return f"{gpu_type}:{count}"


def _cloud_field(cloud: str) -> str:
    return _CLOUD_ALIASES.get(cloud, cloud)


def argv_to_command(module: str, argv: list[str]) -> str:
    parts = ["python", "-m", module, *argv]
    return " ".join(shlex.quote(part) for part in parts)


def build_cluster_placement(plan: DeploymentPlan) -> ClusterPlacement:
    _validate_disagg_cluster_plan(plan)
    return ClusterPlacement(
        prefill_ranks=[0],
        decode_ranks=list(range(1, 1 + plan.decode.replicas)),
    )


def build_worker_task_yaml(
    *,
    plan: DeploymentPlan,
    pool: PoolPlan,
    mode: str,
    replica_index: int,
    session_id: str,
) -> str:
    from warply.engines.sglang import SGLangAdapter

    port = pool.base_port + replica_index
    worker = SGLangAdapter().render_worker(plan=plan, pool=pool, mode=mode, port=port)
    command = argv_to_command(str(worker["module"]), list(worker["argv"]))
    name = cluster_name(session_id=session_id, role=mode, index=replica_index)
    task = {
        "name": name,
        "resources": {
            "cloud": _cloud_field(plan.cloud),
            "accelerators": accelerator_spec(pool.gpu_type, pool.gpus_per_replica),
        },
        "setup": _WORKER_SETUP,
        "run": command,
    }
    return dump_yaml(task)


def build_router_task_yaml(
    *,
    plan: DeploymentPlan,
    prefill_url: str,
    decode_url: str,
    session_id: str,
) -> str:
    from warply.engines.sglang import SGLangAdapter

    routing_plan = replace_routing_urls(plan, prefill_url=prefill_url, decode_url=decode_url)
    router = SGLangAdapter().render_router(routing_plan)
    command = argv_to_command(str(router["module"]), list(router["argv"]))
    name = router_cluster_name(session_id=session_id)
    task = {
        "name": name,
        "resources": {
            "cloud": _cloud_field(plan.cloud),
            "accelerators": accelerator_spec(plan.decode.gpu_type, 1),
        },
        "setup": _ROUTER_SETUP,
        "run": command,
    }
    return dump_yaml(task)


def build_disagg_cluster_task_yaml(*, plan: DeploymentPlan, session_id: str) -> str:
    """Render one multi-node SkyPilot task for v0 prefill/decode disagg."""
    placement = build_cluster_placement(plan)

    from warply.engines.sglang import SGLangAdapter

    adapter = SGLangAdapter()
    prefill = adapter.render_worker(
        plan=plan,
        pool=plan.prefill,
        mode="prefill",
        port=plan.prefill.base_port,
    )
    decode = adapter.render_worker(
        plan=plan,
        pool=plan.decode,
        mode="decode",
        port=plan.decode.base_port,
    )
    router = adapter.render_router(
        plan,
        prefill_urls=["WARPLY_PREFILL_URL"],
        decode_urls=["WARPLY_DECODE_URLS"],
    )

    prefill_command = argv_to_command(str(prefill["module"]), list(prefill["argv"]))
    decode_command = argv_to_command(str(decode["module"]), list(decode["argv"]))
    router_command = argv_to_command(str(router["module"]), list(router["argv"]))
    router_command = router_command.replace("WARPLY_PREFILL_URL", '"$PREFILL_URL"')
    router_command = router_command.replace("WARPLY_DECODE_URL", '"$DECODE_URL"')

    run = f"""\
set -euo pipefail
{_WAIT_HTTP_SCRIPT}
RANK="$SKYPILOT_NODE_RANK"
PREFILL_HOST="$(echo "$SKYPILOT_NODE_IPS" | sed -n '1p')"
PREFILL_URL="http://${{PREFILL_HOST}}:{plan.prefill.base_port}"
DECODE_URLS=""
for DECODE_RANK in $(seq 1 {plan.decode.replicas}); do
  DECODE_LINE=$((DECODE_RANK + 1))
  DECODE_HOST="$(echo "$SKYPILOT_NODE_IPS" | sed -n "${{DECODE_LINE}}p")"
  DECODE_URL="http://${{DECODE_HOST}}:{plan.decode.base_port}"
  if [ -z "$DECODE_URLS" ]; then
    DECODE_URLS="$DECODE_URL"
  else
    DECODE_URLS="${{DECODE_URLS}},${{DECODE_URL}}"
  fi
done

if [ "$RANK" = "0" ]; then
  {prefill_command} &
  PREFILL_PID=$!
  wait_for_http "${{PREFILL_URL}}/health" "prefill" "$PREFILL_PID"
  OLD_IFS="$IFS"
  IFS=","
  for DECODE_URL in $DECODE_URLS; do
    wait_for_http "${{DECODE_URL}}/health" "decode"
  done
  IFS="$OLD_IFS"
  {router_command}
elif [ "$RANK" -ge "1" ] && [ "$RANK" -le "{plan.decode.replicas}" ]; then
  {decode_command}
else
  echo "unsupported SKYPILOT_NODE_RANK=${{RANK}}" >&2
  exit 1
fi
"""
    task = {
        "name": disagg_cluster_name(session_id=session_id),
        "resources": {
            "cloud": _cloud_field(plan.cloud),
            "accelerators": accelerator_spec(
                plan.prefill.gpu_type,
                plan.prefill.gpus_per_replica,
            ),
            "num_nodes": placement.num_nodes,
            "network_tier": "best",
        },
        "setup": _WORKER_SETUP,
        "run": run,
    }
    return dump_yaml(task)


def _validate_disagg_cluster_plan(plan: DeploymentPlan) -> None:
    from warply.exceptions import ValidationError

    if plan.prefill.accelerator.runtime == "rocm" or plan.decode.accelerator.runtime == "rocm":
        raise ValidationError(
            "AMD/ROCm cloud launch is planned but not enabled yet; "
            "export_yaml() works for inspection."
        )
    if plan.prefill.accelerator.runtime != "cuda" or plan.decode.accelerator.runtime != "cuda":
        raise ValidationError("SkyPilot disagg cluster v0 requires known CUDA GPU profiles.")
    if plan.prefill.replicas != 1:
        raise ValidationError(
            "SkyPilot disagg cluster v0 requires prefill replicas=1. "
            "Use decode replicas for 1:N scaling."
        )
    if plan.prefill.gpu_type != plan.decode.gpu_type:
        raise ValidationError("SkyPilot disagg cluster v0 requires matching GPU types.")
    if plan.prefill.gpus_per_replica != plan.decode.gpus_per_replica:
        raise ValidationError("SkyPilot disagg cluster v0 requires matching GPU counts per node.")


def replace_routing_urls(
    plan: DeploymentPlan,
    *,
    prefill_url: str,
    decode_url: str,
) -> DeploymentPlan:
    from dataclasses import replace

    from warply.compiler.plan import RoutingConfig

    routing = RoutingConfig(
        mode=plan.routing.mode,
        router_port=plan.routing.router_port,
        endpoint=plan.routing.endpoint,
        prefill_base_url=prefill_url,
        decode_base_url=decode_url,
    )
    return replace(plan, routing=routing)

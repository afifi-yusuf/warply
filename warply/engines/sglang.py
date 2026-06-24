from __future__ import annotations

from warply.compiler.plan import DeploymentPlan, PoolPlan
from warply.exceptions import ValidationError


class SGLangAdapter:
    """Render SGLang prefill/decode/router process configs."""

    def render_prefill(self, plan: DeploymentPlan) -> dict[str, object]:
        return self.render_worker(plan=plan, pool=plan.prefill, mode="prefill")

    def render_decode(self, plan: DeploymentPlan) -> dict[str, object]:
        return self.render_worker(plan=plan, pool=plan.decode, mode="decode")

    def render_router(
        self,
        plan: DeploymentPlan,
        *,
        prefill_urls: list[str] | None = None,
        decode_urls: list[str] | None = None,
    ) -> dict[str, object]:
        prefill_target = ",".join(prefill_urls) if prefill_urls else plan.routing.prefill_base_url
        decode_target = ",".join(decode_urls) if decode_urls else plan.routing.decode_base_url
        return {
            "name": "sglang-router",
            "module": "sglang_router.launch_router",
            "argv": [
                "--pd-disaggregation",
                "--host",
                "0.0.0.0",
                "--port",
                str(plan.routing.router_port),
                "--prefill",
                prefill_target,
                "--decode",
                decode_target,
            ],
            "env": {},
            "port": plan.routing.router_port,
        }

    def health(self, pool: PoolPlan) -> bool:
        return pool.replicas > 0

    def openai_base_url(self, plan: DeploymentPlan) -> str:
        return plan.routing.endpoint

    def render_worker(
        self,
        *,
        plan: DeploymentPlan,
        pool: PoolPlan,
        mode: str,
        port: int | None = None,
    ) -> dict[str, object]:
        worker_port = port if port is not None else pool.base_port
        if plan.speculation.enabled:
            raise ValidationError(
                "Speculative decoding is planned but not enabled for the SGLang adapter yet."
            )
        kv_transfer = plan.resolved_kv_transfer
        if kv_transfer is None:
            raise ValidationError(
                "No supported SGLang KV transfer is resolved for this accelerator profile."
            )
        return {
            "name": f"sglang-{mode}",
            "module": "sglang.launch_server",
            "argv": [
                "--model-path",
                plan.model,
                "--host",
                "0.0.0.0",
                "--port",
                str(worker_port),
                "--tp-size",
                str(pool.gpus_per_replica),
                "--disaggregation-mode",
                mode,
                "--disaggregation-transfer-backend",
                kv_transfer,
            ],
            "env": {},
            "port": worker_port,
            "replicas": pool.replicas,
        }

    def _render_worker(
        self,
        *,
        plan: DeploymentPlan,
        pool: PoolPlan,
        mode: str,
    ) -> dict[str, object]:
        return self.render_worker(plan=plan, pool=pool, mode=mode)

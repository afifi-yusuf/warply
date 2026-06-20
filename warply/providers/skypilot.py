from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING

from warply.providers.base import Node
from warply.providers.skypilot_task import (
    build_disagg_cluster_task_yaml,
    disagg_cluster_name,
)
from warply.runtime.health import wait_for_http_ready
from warply.runtime.routing import node_http_url

if TYPE_CHECKING:
    from warply.compiler.plan import DeploymentPlan


@dataclass(frozen=True)
class ClusterLaunch:
    """Resolved hosts for a launched disagg cluster."""

    cluster_name: str
    router_host: str
    prefill_hosts: list[str]
    decode_hosts: list[str]


class SkyPilotProvider:
    """Provision Warply pools on cloud infrastructure via SkyPilot."""

    def __init__(self, *, plan: DeploymentPlan, session_id: str | None = None) -> None:
        self.plan = plan
        self.session_id = session_id or uuid.uuid4().hex[:8]
        self._router_node: Node | None = None
        self._downed_clusters: set[str] = set()

    def provision_cluster(self) -> tuple[list[Node], list[Node], Node]:
        name = disagg_cluster_name(session_id=self.session_id)
        yaml_str = build_disagg_cluster_task_yaml(plan=self.plan, session_id=self.session_id)
        launch = self._launch_cluster(yaml_str=yaml_str, cluster_name=name)
        try:
            self._wait_for_router_ready(launch.router_host)
        except Exception:
            self._down_cluster(name)
            raise

        prefill = Node(
            id=f"{name}-prefill-0",
            role="prefill",
            host=launch.prefill_hosts[0],
            port=self.plan.prefill.base_port,
            cluster_name=name,
            healthy=True,
        )
        decode_nodes = [
            Node(
                id=f"{name}-decode-{index}",
                role="decode",
                host=host,
                port=self.plan.decode.base_port,
                cluster_name=name,
                healthy=True,
            )
            for index, host in enumerate(launch.decode_hosts)
        ]
        router = Node(
            id=f"{name}-router",
            role="router",
            host=launch.router_host,
            port=self.plan.routing.router_port,
            cluster_name=name,
            healthy=True,
        )
        self._router_node = router
        return [prefill], decode_nodes, router

    def teardown(self, nodes: list[Node]) -> None:
        for node in nodes:
            if node.role == "router":
                self._router_node = None
            if node.cluster_name in self._downed_clusters:
                continue
            self._downed_clusters.add(node.cluster_name)
            self._down_cluster(node.cluster_name)

    def status(self, nodes: list[Node]) -> list[Node]:
        return nodes

    def _launch_cluster(self, *, yaml_str: str, cluster_name: str) -> ClusterLaunch:
        if os.environ.get("WARPLY_SKYPILOT_DRY_RUN") == "1":
            router_host = f"dryrun.{cluster_name}.example"
            return ClusterLaunch(
                cluster_name=cluster_name,
                router_host=router_host,
                prefill_hosts=[f"dryrun.{cluster_name}.prefill.example"],
                decode_hosts=[
                    f"dryrun.{cluster_name}.decode-{index}.example"
                    for index in range(self.plan.decode.replicas)
                ],
            )

        sky = self._import_sky()
        task = sky.Task.from_yaml_str(yaml_str)
        request_id = sky.launch(task, cluster_name=cluster_name)
        _, handle = sky.stream_and_get(request_id)
        router_host, prefill_hosts, decode_hosts = self._resolve_cluster_hosts(
            handle,
            decode_replicas=self.plan.decode.replicas,
        )
        return ClusterLaunch(
            cluster_name=cluster_name,
            router_host=router_host,
            prefill_hosts=prefill_hosts,
            decode_hosts=decode_hosts,
        )

    def _wait_for_router_ready(self, router_host: str) -> None:
        if os.environ.get("WARPLY_SKYPILOT_DRY_RUN") == "1":
            return

        timeout = float(os.environ.get("WARPLY_ROUTER_READY_TIMEOUT", "600"))
        router_url = node_http_url(
            Node(
                id="router",
                role="router",
                host=router_host,
                port=self.plan.routing.router_port,
            )
        )
        try:
            wait_for_http_ready(router_url, timeout=timeout)
        except TimeoutError as exc:
            raise RuntimeError(f"router not ready at {router_url}") from exc

    def _down_cluster(self, cluster_name: str) -> None:
        if not cluster_name:
            return
        if os.environ.get("WARPLY_SKYPILOT_DRY_RUN") == "1":
            return

        sky = self._import_sky()
        sky.down(cluster_name)

    @staticmethod
    def _import_sky():
        try:
            import sky
        except ImportError as exc:
            raise ImportError(
                "SkyPilot is required for cloud provisioning. "
                "Install with: pip install warply[cloud]"
            ) from exc
        return sky

    @staticmethod
    def _resolve_cluster_hosts(
        handle,
        *,
        decode_replicas: int,
    ) -> tuple[str, list[str], list[str]]:
        """Return public router host plus internal prefill/decode hosts by rank."""
        router_host = SkyPilotProvider._resolve_head_ip(handle)
        node_ips = SkyPilotProvider._collect_node_ips(handle, fallback_head=router_host)
        required_nodes = 1 + decode_replicas

        if len(node_ips) >= required_nodes:
            return router_host, [node_ips[0]], node_ips[1:required_nodes]

        raise RuntimeError(
            f"could not resolve {required_nodes} node IPs for SkyPilot cluster "
            f"{handle.get_cluster_name()!r}; got {len(node_ips)}"
        )

    @staticmethod
    def _collect_node_ips(handle, *, fallback_head: str) -> list[str]:
        if hasattr(handle, "internal_ips"):
            raw = handle.internal_ips()
            if raw:
                return [str(ip) for ip in raw if ip]

        ips: list[str] = []
        if hasattr(handle, "external_ips"):
            raw = handle.external_ips()
            if raw:
                for ip in raw:
                    value = str(ip)
                    if value and value not in ips:
                        ips.append(value)

        if fallback_head and fallback_head not in ips:
            ips.append(fallback_head)

        return ips

    @staticmethod
    def _resolve_head_ip(handle) -> str:
        if hasattr(handle, "head_ip") and handle.head_ip:
            return str(handle.head_ip)

        if hasattr(handle, "external_ips"):
            ips = handle.external_ips()
            if ips:
                return str(ips[0])

        cluster_name = handle.get_cluster_name()
        raise RuntimeError(f"could not resolve external IP for SkyPilot cluster {cluster_name!r}")

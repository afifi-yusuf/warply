from __future__ import annotations

from dataclasses import dataclass, field

from warply.compiler.plan import DeploymentPlan
from warply.providers.base import Node, ProviderPlugin
from warply.providers.local_mock import LocalMockProvider
from warply.router.mock import MockRouter
from warply.runtime.client import HTTPOpenAIClient, MockOpenAIClient
from warply.runtime.routing import node_http_url, primary_pool_url, resolve_routing
from warply.types import EngineState


@dataclass
class Runtime:
    """State machine that binds a plan to provider/router plugins."""

    plan: DeploymentPlan
    provider: ProviderPlugin = field(default_factory=LocalMockProvider)
    router: MockRouter = field(default_factory=MockRouter)
    prefill_nodes: list[Node] = field(default_factory=list)
    decode_nodes: list[Node] = field(default_factory=list)
    router_nodes: list[Node] = field(default_factory=list)
    endpoint: str | None = None

    def up(self) -> None:
        try:
            self.prefill_nodes = self.provider.provision(self.plan.prefill.provision)
            self.decode_nodes = self.provider.provision(self.plan.decode.provision)
            if self.plan.cloud == "local":
                router = self.provider.provision_router(
                    prefill_url="",
                    decode_url="",
                )
                self.router_nodes = [router]
                self.endpoint = node_http_url(router)
                return

            prefill_url = primary_pool_url(self.prefill_nodes)
            decode_url = primary_pool_url(self.decode_nodes)
            router = self.provider.provision_router(
                prefill_url=prefill_url,
                decode_url=decode_url,
            )
            self.router_nodes = [router]
            self.plan = resolve_routing(
                self.plan,
                prefill_nodes=self.prefill_nodes,
                decode_nodes=self.decode_nodes,
                router_node=router,
            )
            self.endpoint = self.plan.routing.endpoint
        except Exception:
            self.down()
            raise

    def scale(
        self,
        *,
        plan: DeploymentPlan,
        prefill: int | None = None,
        decode: int | None = None,
    ) -> None:
        next_prefill_nodes = self.prefill_nodes
        next_decode_nodes = self.decode_nodes
        provisioned_nodes: list[Node] = []

        if prefill is not None:
            next_prefill_nodes = self.provider.provision(plan.prefill.provision)
            provisioned_nodes.extend(next_prefill_nodes)
        try:
            if decode is not None:
                next_decode_nodes = self.provider.provision(plan.decode.provision)
                provisioned_nodes.extend(next_decode_nodes)
        except Exception:
            for node in provisioned_nodes:
                self.provider.teardown([node])
            raise

        if prefill is not None:
            self.provider.teardown(self.prefill_nodes)
            self.prefill_nodes = next_prefill_nodes
        if decode is not None:
            self.provider.teardown(self.decode_nodes)
            self.decode_nodes = next_decode_nodes
        self.plan = plan

    def down(self) -> None:
        self.provider.teardown(self.prefill_nodes)
        self.provider.teardown(self.decode_nodes)
        self.provider.teardown(self.router_nodes)
        self.prefill_nodes = []
        self.decode_nodes = []
        self.router_nodes = []
        self.endpoint = None

    def client(self) -> HTTPOpenAIClient | MockOpenAIClient:
        base_url = self.endpoint or self.plan.routing.endpoint
        if self.plan.cloud == "local":
            return MockOpenAIClient(base_url=base_url)
        return HTTPOpenAIClient(base_url=base_url)

    def healthy_prefill(self) -> int:
        return sum(node.healthy for node in self.provider.status(self.prefill_nodes))

    def healthy_decode(self) -> int:
        return sum(node.healthy for node in self.provider.status(self.decode_nodes))


def state_after_down(current: EngineState) -> EngineState:
    stoppable_states = {
        EngineState.PENDING,
        EngineState.READY,
        EngineState.SCALING,
        EngineState.STOPPING,
    }
    if current in stoppable_states:
        return EngineState.STOPPED
    return current

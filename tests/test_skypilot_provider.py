from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import ValidationError
from warply.providers.skypilot import ClusterLaunch, SkyPilotProvider
from warply.runtime.client import MockOpenAIClient
from warply.types import EngineState


def make_engine(**overrides) -> wp.DisaggEngine:
    kwargs = dict(
        model="meta-llama/Llama-3.1-8B",
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
        cloud="lambda",
    )
    kwargs.update(overrides)
    return wp.DisaggEngine(**kwargs)


def test_skypilot_provider_rejects_multireplica_cluster():
    plan = make_engine(prefill=wp.Pool("1xH100", replicas=2)).plan()
    provider = SkyPilotProvider(plan=plan, session_id="test1234")

    with pytest.raises(ValidationError, match="prefill replicas=1"):
        provider.provision_cluster()


def test_lambda_up_dry_run_with_decode_scaling(monkeypatch):
    monkeypatch.setenv("WARPLY_SKYPILOT_DRY_RUN", "1")
    engine = make_engine(decode=wp.Pool("1xH100", replicas=3))

    engine.up()
    status = engine.status()

    assert status.state is EngineState.READY
    assert status.endpoint.startswith("http://dryrun.warply-")
    assert "disagg" in status.endpoint
    assert status.prefill.healthy_replicas == 1
    assert status.decode.healthy_replicas == 3
    assert engine.deployed_plan() is not None
    assert engine.deployed_plan().routing.prefill_base_url == "warply://lambda/prefill"
    assert engine.deployed_plan().routing.decode_base_url == "warply://lambda/decode"
    assert isinstance(engine.client(), MockOpenAIClient)
    assert engine.client().base_url == status.endpoint
    assert engine.generate("hello").endswith("hello")

    assert engine._runtime is not None
    provider = engine._runtime.provider
    prefill_host = engine._runtime.prefill_nodes[0].host
    decode_hosts = [node.host for node in engine._runtime.decode_nodes]
    router_host = engine._runtime.router_nodes[0].host
    assert prefill_host not in decode_hosts
    assert "prefill" in prefill_host
    assert len(decode_hosts) == 3
    assert all("decode" in host for host in decode_hosts)
    assert router_host == status.endpoint.removeprefix("http://").split(":")[0]

    cluster_names = {
        node.cluster_name
        for node in [
            *engine._runtime.prefill_nodes,
            *engine._runtime.decode_nodes,
            *engine._runtime.router_nodes,
        ]
    }
    assert cluster_names == {"warply-" + engine._runtime.provider.session_id + "-disagg"}

    engine.down()
    assert engine.status().state is EngineState.STOPPED
    assert engine._runtime is None
    assert provider._downed_clusters == cluster_names


def test_lambda_scale_not_implemented():
    engine = make_engine()
    engine._state = EngineState.READY
    engine._runtime = object()

    with pytest.raises(NotImplementedError, match="cloud scale"):
        engine.scale(decode=2)


def test_cloud_up_failure_teardowns_cluster(monkeypatch):
    monkeypatch.delenv("WARPLY_SKYPILOT_DRY_RUN", raising=False)
    launched: list[str] = []
    downed: list[str] = []

    def fake_launch(self, *, yaml_str: str, cluster_name: str) -> ClusterLaunch:
        launched.append(cluster_name)
        return ClusterLaunch(
            cluster_name=cluster_name,
            router_host="10.0.0.1",
            prefill_hosts=["10.0.0.1"],
            decode_hosts=["10.0.0.2"],
        )

    def fail_ready(self, router_host: str) -> None:
        raise RuntimeError("router not ready")

    def track_down(self, cluster_name: str) -> None:
        downed.append(cluster_name)

    monkeypatch.setattr(SkyPilotProvider, "_launch_cluster", fake_launch)
    monkeypatch.setattr(SkyPilotProvider, "_wait_for_router_ready", fail_ready)
    monkeypatch.setattr(SkyPilotProvider, "_down_cluster", track_down)

    engine = make_engine()
    with pytest.raises(RuntimeError, match="router not ready"):
        engine.up()

    assert len(launched) == 1
    assert launched[0].endswith("-disagg")
    assert downed == [launched[0]]
    assert engine.status().state is EngineState.STOPPED
    assert engine._runtime is None


def test_resolve_cluster_hosts_uses_internal_ips():
    class _Handle:
        head_ip = "203.0.113.10"

        @staticmethod
        def internal_ips():
            return ["10.0.0.1", "10.0.0.2"]

        @staticmethod
        def external_ips():
            return ["203.0.113.10"]

        @staticmethod
        def get_cluster_name():
            return "warply-test-disagg"

    router_host, prefill_hosts, decode_hosts = SkyPilotProvider._resolve_cluster_hosts(
        _Handle(),
        decode_replicas=1,
    )
    assert router_host == "203.0.113.10"
    assert prefill_hosts == ["10.0.0.1"]
    assert decode_hosts == ["10.0.0.2"]


def test_resolve_cluster_hosts_preserves_internal_rank_order_without_external_ips():
    class _Handle:
        head_ip = "203.0.113.10"

        @staticmethod
        def internal_ips():
            return ["10.0.0.1", "10.0.0.2"]

        @staticmethod
        def get_cluster_name():
            return "warply-test-disagg"

    router_host, prefill_hosts, decode_hosts = SkyPilotProvider._resolve_cluster_hosts(
        _Handle(),
        decode_replicas=1,
    )
    assert router_host == "203.0.113.10"
    assert prefill_hosts == ["10.0.0.1"]
    assert decode_hosts == ["10.0.0.2"]


def test_resolve_cluster_hosts_returns_all_decode_hosts():
    class _Handle:
        head_ip = "203.0.113.10"

        @staticmethod
        def internal_ips():
            return ["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4"]

        @staticmethod
        def get_cluster_name():
            return "warply-test-disagg"

    router_host, prefill_hosts, decode_hosts = SkyPilotProvider._resolve_cluster_hosts(
        _Handle(),
        decode_replicas=3,
    )
    assert router_host == "203.0.113.10"
    assert prefill_hosts == ["10.0.0.1"]
    assert decode_hosts == ["10.0.0.2", "10.0.0.3", "10.0.0.4"]


def test_resolve_cluster_hosts_rejects_too_few_internal_ips():
    class _Handle:
        head_ip = "203.0.113.10"

        @staticmethod
        def internal_ips():
            return ["10.0.0.1", "10.0.0.2"]

        @staticmethod
        def get_cluster_name():
            return "warply-test-disagg"

    with pytest.raises(RuntimeError, match="could not resolve 4 node IPs"):
        SkyPilotProvider._resolve_cluster_hosts(_Handle(), decode_replicas=3)

from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import ValidationError
from warply.providers.skypilot_task import (
    build_cluster_placement,
    build_disagg_cluster_task_yaml,
    build_router_task_yaml,
    build_worker_task_yaml,
    cluster_name,
    disagg_cluster_name,
)


def make_plan(**overrides):
    kwargs = dict(
        model="meta-llama/Llama-3.1-8B",
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
        cloud="lambda",
    )
    kwargs.update(overrides)
    return wp.DisaggEngine(**kwargs).plan()


def test_worker_task_yaml_contains_sglang_and_lambda():
    plan = make_plan()
    yaml_str = build_worker_task_yaml(
        plan=plan,
        pool=plan.prefill,
        mode="prefill",
        replica_index=0,
        session_id="abc12345",
    )

    assert cluster_name(session_id="abc12345", role="prefill", index=0) in yaml_str
    assert "cloud: lambda" in yaml_str
    assert "H100:1" in yaml_str
    assert "sglang.launch_server" in yaml_str
    assert "--disaggregation-mode" in yaml_str
    assert "prefill" in yaml_str
    assert "--disaggregation-transfer-backend" in yaml_str
    assert "nixl" in yaml_str


def test_router_task_yaml_uses_resolved_urls():
    plan = make_plan()
    yaml_str = build_router_task_yaml(
        plan=plan,
        prefill_url="http://10.0.0.1:31000",
        decode_url="http://10.0.0.2:32000",
        session_id="abc12345",
    )

    assert "sglang_router.launch_router" in yaml_str
    assert "--pd-disaggregation" in yaml_str
    assert "http://10.0.0.1:31000" in yaml_str
    assert "http://10.0.0.2:32000" in yaml_str


def test_cluster_placement_maps_prefill_and_decode_ranks():
    plan = make_plan(decode=wp.Pool("1xH100", replicas=3))
    placement = build_cluster_placement(plan)

    assert placement.prefill_ranks == [0]
    assert placement.decode_ranks == [1, 2, 3]
    assert placement.num_nodes == 4


def test_disagg_cluster_task_yaml_uses_multinode_best_network():
    plan = make_plan(decode=wp.Pool("1xH100", replicas=3))
    yaml_str = build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")

    assert disagg_cluster_name(session_id="abc12345") in yaml_str
    assert "cloud: lambda" in yaml_str
    assert "accelerators: 'H100:1'" in yaml_str
    assert "num_nodes: 4" in yaml_str
    assert "network_tier: best" in yaml_str
    assert "SKYPILOT_NODE_RANK" in yaml_str
    assert "SKYPILOT_NODE_IPS" in yaml_str
    assert "DECODE_URLS" in yaml_str
    assert "seq 1 3" in yaml_str
    assert "wait_for_http" in yaml_str
    assert "PREFILL_PID" in yaml_str
    assert "${PREFILL_URL}/health" in yaml_str
    assert "${DECODE_URL}/health" in yaml_str
    assert "sglang.launch_server" in yaml_str
    assert "--disaggregation-mode prefill" in yaml_str
    assert "--disaggregation-mode decode" in yaml_str
    assert "sglang_router.launch_router" in yaml_str
    assert "--disaggregation-transfer-backend nixl" in yaml_str
    assert yaml_str.count("accelerators:") == 1


def test_disagg_cluster_task_rejects_multireplica_prefill_cloud_plan():
    plan = make_plan(prefill=wp.Pool("1xH100", replicas=2))

    with pytest.raises(ValidationError, match="prefill replicas=1"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")


def test_disagg_cluster_task_rejects_mismatched_gpu_type():
    plan = make_plan(decode=wp.Pool("1xA100", replicas=1))

    with pytest.raises(ValidationError, match="GPU types"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")


def test_disagg_cluster_task_rejects_mismatched_gpu_count():
    plan = make_plan(decode=wp.Pool("2xH100", replicas=1))

    with pytest.raises(ValidationError, match="GPU counts"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")


def test_disagg_cluster_task_rejects_amd_rocm_until_live_support_lands():
    plan = make_plan(
        prefill=wp.Pool("1xMI300X", replicas=1),
        decode=wp.Pool("1xMI300X", replicas=1),
        kv_transfer="auto",
    )

    with pytest.raises(ValidationError, match="AMD/ROCm cloud launch"):
        build_disagg_cluster_task_yaml(plan=plan, session_id="abc12345")

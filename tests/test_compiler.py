from __future__ import annotations

import warply as wp
from warply.compiler import compile
from warply.engines.sglang import SGLangAdapter
from warply.kv.nixl import NixlTransfer


def make_engine(**overrides) -> wp.DisaggEngine:
    kwargs = dict(
        model="meta-llama/Llama-3.1-70B",
        prefill=wp.Pool("4xH100", replicas=2),
        decode=wp.Pool("2xH100", replicas=4),
        cloud="local",
    )
    kwargs.update(overrides)
    return wp.DisaggEngine(**kwargs)


def test_compile_plan_is_deterministic():
    plan = compile(make_engine())

    assert plan.to_dict() == {
        "model": "meta-llama/Llama-3.1-70B",
        "backend": "sglang",
        "kv_transfer": "nixl",
        "cloud": "local",
        "prefill": {
            "role": "prefill",
            "gpus": "4xH100",
            "gpu_type": "H100",
            "gpus_per_replica": 4,
            "replicas": 2,
            "base_port": 31000,
            "provision": {
                "role": "prefill",
                "cloud": "local",
                "gpu_type": "H100",
                "gpus_per_replica": 4,
                "replicas": 2,
            },
        },
        "decode": {
            "role": "decode",
            "gpus": "2xH100",
            "gpu_type": "H100",
            "gpus_per_replica": 2,
            "replicas": 4,
            "base_port": 32000,
            "provision": {
                "role": "decode",
                "cloud": "local",
                "gpu_type": "H100",
                "gpus_per_replica": 2,
                "replicas": 4,
            },
        },
        "routing": {
            "mode": "prefill_decode",
            "router_port": 8000,
            "endpoint": "http://127.0.0.1:8000",
            "prefill_base_url": "http://127.0.0.1:31000",
            "decode_base_url": "http://127.0.0.1:32000",
        },
    }


def test_sglang_adapter_renders_prefill_decode_and_router():
    plan = compile(make_engine())
    adapter = SGLangAdapter()

    prefill = adapter.render_prefill(plan)
    decode = adapter.render_decode(plan)
    router = adapter.render_router(plan)

    assert "--disaggregation-mode" in prefill["argv"]
    assert "prefill" in prefill["argv"]
    assert "--tp-size" in decode["argv"]
    assert "2" in decode["argv"]
    assert "--pd-disaggregation" in router["argv"]
    assert adapter.openai_base_url(plan) == "http://127.0.0.1:8000"


def test_sglang_router_renders_multiple_decode_targets():
    plan = compile(make_engine())
    router = SGLangAdapter().render_router(
        plan,
        prefill_urls=["http://10.0.0.1:31000"],
        decode_urls=["http://10.0.0.2:32000", "http://10.0.0.3:32000"],
    )

    assert router["argv"][router["argv"].index("--prefill") + 1] == "http://10.0.0.1:31000"
    assert (
        router["argv"][router["argv"].index("--decode") + 1]
        == "http://10.0.0.2:32000,http://10.0.0.3:32000"
    )


def test_nixl_config_renders_transfer_settings():
    plan = compile(make_engine())
    config = NixlTransfer().configure(plan)

    assert config["backend"] == "nixl"
    assert "--disaggregation-transfer-backend" in config["argv"]
    assert config["env"]["WARPLY_KV_TRANSFER"] == "nixl"


def test_export_yaml_contains_plan_sections():
    yaml = make_engine().export_yaml()

    assert "model: meta-llama/Llama-3.1-70B" in yaml
    assert "prefill:" in yaml
    assert "decode:" in yaml
    assert "routing:" in yaml


def test_plan_reflects_scaled_replicas():
    engine = make_engine()

    engine.up()
    engine.scale(decode=3)

    assert engine.status().decode.replicas == 3
    assert engine.plan().decode.replicas == 3
    assert engine.plan().decode.provision.replicas == 3
    assert "replicas: 3" in engine.export_yaml()


def test_non_local_plan_uses_provider_placeholders():
    plan = compile(make_engine(cloud="lambda"))

    assert not plan.routing.endpoint.startswith("http://127.0.0.1")
    assert plan.routing.endpoint == "warply://lambda/router"
    assert plan.routing.prefill_base_url == "warply://lambda/prefill"

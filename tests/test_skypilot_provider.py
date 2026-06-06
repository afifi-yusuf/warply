from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import ValidationError
from warply.providers.skypilot import SkyPilotProvider
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


def test_skypilot_provider_rejects_multireplica_pool():
    plan = make_engine(prefill=wp.Pool("1xH100", replicas=2)).plan()
    provider = SkyPilotProvider(plan=plan, session_id="test1234")

    with pytest.raises(ValidationError, match="replicas=1"):
        provider.provision(plan.prefill.provision)


def test_lambda_up_dry_run(monkeypatch):
    monkeypatch.setenv("WARPLY_SKYPILOT_DRY_RUN", "1")
    engine = make_engine()

    engine.up()
    status = engine.status()

    assert status.state is EngineState.READY
    assert status.endpoint.startswith("http://dryrun.warply-")
    assert status.prefill.healthy_replicas == 1
    assert status.decode.healthy_replicas == 1
    assert engine.deployed_plan() is not None
    assert engine.deployed_plan().routing.prefill_base_url.startswith("http://dryrun.")
    assert engine.deployed_plan().routing.decode_base_url.startswith("http://dryrun.")
    assert isinstance(engine.client(), MockOpenAIClient)
    assert engine.client().base_url == status.endpoint
    assert engine.generate("hello").endswith("hello")

    engine.down()
    assert engine.status().state is EngineState.STOPPED


def test_lambda_scale_not_implemented():
    engine = make_engine()
    engine._state = EngineState.READY
    engine._runtime = object()

    with pytest.raises(NotImplementedError, match="cloud scale"):
        engine.scale(decode=2)

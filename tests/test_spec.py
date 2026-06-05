"""Spec-layer contract tests.

These cover the only part of the SDK that is implemented today: spec validation,
status reporting, and lifecycle guards. They lock the public contract before the
compiler/provider/engine layers land.
"""

from __future__ import annotations

import pytest

import warply as wp
from warply.exceptions import NotReadyError, ValidationError
from warply.types import EngineState


def make_engine(**overrides) -> wp.DisaggEngine:
    kwargs = dict(
        model="meta-llama/Llama-3.1-70B",
        prefill=wp.Pool("4xH100", replicas=2),
        decode=wp.Pool("2xH100", replicas=4),
        cloud="local",
    )
    kwargs.update(overrides)
    return wp.DisaggEngine(**kwargs)


class TestPool:
    def test_valid_pool(self):
        pool = wp.Pool("4xH100", replicas=2)
        assert pool.gpus == "4xH100"
        assert pool.replicas == 2

    def test_default_replicas(self):
        assert wp.Pool("1xA100").replicas == 1

    @pytest.mark.parametrize("bad", ["H100", "4H100", "xH100", "4x", "4 x H100", ""])
    def test_invalid_gpu_spec(self, bad):
        with pytest.raises(ValidationError):
            wp.Pool(bad)

    def test_zero_gpu_count_rejected(self):
        with pytest.raises(ValidationError):
            wp.Pool("0xH100")

    def test_zero_replicas_rejected(self):
        with pytest.raises(ValidationError):
            wp.Pool("4xH100", replicas=0)


class TestEngineValidation:
    def test_defaults(self):
        engine = make_engine()
        assert engine.backend == "sglang"
        assert engine.kv_transfer == "nixl"

    def test_empty_model_rejected(self):
        with pytest.raises(ValidationError):
            make_engine(model="   ")

    def test_unsupported_backend(self):
        with pytest.raises(ValidationError):
            make_engine(backend="vllm")

    def test_unsupported_kv_transfer(self):
        with pytest.raises(ValidationError):
            make_engine(kv_transfer="rdma")

    def test_unsupported_cloud(self):
        with pytest.raises(ValidationError):
            make_engine(cloud="gcp")


class TestStatus:
    def test_status_reflects_spec(self):
        engine = make_engine()
        status = engine.status()
        assert status.state is EngineState.PENDING
        assert status.prefill.replicas == 2
        assert status.decode.replicas == 4
        assert status.endpoint is None

    def test_status_to_dict(self):
        d = make_engine().status().to_dict()
        assert d["state"] == "pending"
        assert d["prefill"]["gpus"] == "4xH100"
        assert d["decode"]["replicas"] == 4


class TestLifecycleGuards:
    def test_scale_requires_argument(self):
        with pytest.raises(ValidationError):
            make_engine().scale()

    @pytest.mark.parametrize("kwargs", [{"prefill": 0}, {"decode": -1}])
    def test_scale_rejects_bad_replicas(self, kwargs):
        with pytest.raises(ValidationError):
            make_engine().scale(**kwargs)

    def test_scale_before_up_is_not_ready(self):
        with pytest.raises(NotReadyError):
            make_engine().scale(decode=8)

    def test_scale_does_not_mutate_on_failure(self):
        engine = make_engine()
        with pytest.raises(NotReadyError):
            engine.scale(decode=99)
        assert engine.status().decode.replicas == 4

    def test_client_before_up_is_not_ready(self):
        with pytest.raises(NotReadyError):
            make_engine().client()

    def test_generate_empty_prompt_rejected(self):
        with pytest.raises(ValidationError):
            make_engine().generate("  ")

    def test_down_before_up_is_idempotent(self):
        engine = make_engine()
        engine.down()
        assert engine.status().state is EngineState.STOPPED


class TestInternalsHidden:
    def test_internal_fields_not_in_init(self):
        with pytest.raises(TypeError):
            wp.DisaggEngine(
                model="m",
                prefill=wp.Pool("1xH100"),
                decode=wp.Pool("1xH100"),
                _state=EngineState.READY,
            )

    def test_internal_fields_not_in_repr(self):
        assert "_state" not in repr(make_engine())

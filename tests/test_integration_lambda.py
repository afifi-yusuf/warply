from __future__ import annotations

import os

import pytest

import warply as wp

pytestmark = pytest.mark.skipif(
    os.environ.get("WARPLY_INTEGRATION") != "1",
    reason="Lambda/SkyPilot integration tests require WARPLY_INTEGRATION=1.",
)


def test_lambda_sglang_e2e():
    """Live Lambda + SkyPilot + SGLang/NIXL end-to-end test.

    Requires:
    - pip install warply[cloud]
    - SkyPilot configured for Lambda (sky check)
    - GPUs available on Lambda
    """
    if os.environ.get("WARPLY_SKYPILOT_DRY_RUN") == "1":
        pytest.skip("WARPLY_SKYPILOT_DRY_RUN=1 disables live SkyPilot launches.")

    engine = wp.DisaggEngine(
        model=os.environ.get("WARPLY_TEST_MODEL", "meta-llama/Llama-3.1-8B"),
        prefill=wp.Pool("1xH100", replicas=1),
        decode=wp.Pool("1xH100", replicas=1),
        cloud="lambda",
    )

    try:
        engine.up()
        response = engine.generate("hello")
        assert isinstance(response, str)
        assert response.strip()
    finally:
        engine.down()

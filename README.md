# Warply

[![CI](https://github.com/afifi-yusuf/warply/actions/workflows/ci.yml/badge.svg)](https://github.com/afifi-yusuf/warply/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](./pyproject.toml)

**Python control plane for inference and self-improving systems.**

Warply turns serving intent into a runnable deployment plan: prefill/decode pools,
SkyPilot provisioning, SGLang launch flags, NIXL KV transfer, router endpoints, and an
OpenAI-compatible client. The goal is to make advanced LLM serving programmable from
`import warply`, without asking every researcher or startup to own Kubernetes, CRDs, or
per-cloud launch glue.

Learn more at [warply.ai](https://warply.ai).

> **Status:** Pre-alpha. Local mock lifecycle, compiler/export, SGLang/NIXL adapters,
> OpenAI-compatible HTTP client, and SkyPilot Lambda dry-run paths are implemented. Live GPU
> integration remains gated and experimental.

## Why Warply?

High-performance inference mechanisms are becoming substrate. Engines and runtimes such as
[vLLM](https://github.com/vllm-project/vllm), [SGLang](https://github.com/sgl-project/sglang),
[Dynamo](https://github.com/ai-dynamo/dynamo), TensorRT-LLM, and llm-d already expose serious
serving capabilities: optimized kernels, disaggregated prefill/decode, KV-aware routing,
continuous batching, and multi-node execution.

Warply's wedge is the **user-facing control plane** above those mechanisms:

- Launch and tear down model-serving systems from Python.
- Compile one declarative spec into provisioning, engine, KV-transfer, and routing plans.
- Scale prefill/decode pools independently as the workload changes.
- Keep cloud provisioning, runtime selection, and client binding behind a small SDK.
- Grow toward rollout, eval, and RL/self-improvement workflows without changing the user's entrypoint.

If you already operate a mature Kubernetes inference platform, Dynamo or llm-d may be the right
runtime. Warply is for the path where you want a Python object to create, inspect, and control
that system across clouds.

## What Works Today

| Area | Current support |
| --- | --- |
| SDK | `DisaggEngine`, `Pool`, `up()`, `down()`, `scale()` for local mock, `client()`, `generate()` |
| Compiler | Deterministic `DeploymentPlan`, `engine.plan()`, `engine.export_yaml()` |
| Engine | SGLang adapter for prefill, decode, and router process configs |
| KV transfer | NIXL for CUDA plans; `kv_transfer="auto"` resolves to NIXL on known CUDA GPUs |
| Cloud | SkyPilot Lambda/CoreWeave provider skeleton; Lambda dry-run and task rendering |
| Placement | One prefill node plus N decode nodes in one SkyPilot multi-node cluster |
| Client | Mock local client plus OpenAI-compatible HTTP client for deployed routers |
| Hardware planning | CUDA and ROCm accelerator profiles; live ROCm launch intentionally disabled |

## Quick Start

Install from source:

```bash
git clone https://github.com/afifi-yusuf/warply.git
cd warply
pip install -e ".[dev]"
```

Run the no-GPU local lifecycle:

```python
import warply as wp

engine = wp.DisaggEngine(
    model="meta-llama/Llama-3.1-8B",
    prefill=wp.Pool("1xH100", replicas=1),
    decode=wp.Pool("1xH100", replicas=1),
    backend="sglang",
    kv_transfer="nixl",
    cloud="local",
)

engine.up()
print(engine.generate("hello"))
print(engine.status())
engine.down()
```

Inspect the compiled plan:

```python
print(engine.plan())
print(engine.export_yaml())
```

## Cloud Dry Run

Use `WARPLY_SKYPILOT_DRY_RUN=1` to exercise the Lambda control path without GPUs, SkyPilot
credentials, or cloud spend:

```bash
WARPLY_SKYPILOT_DRY_RUN=1 python - <<'PY'
import warply as wp

engine = wp.DisaggEngine(
    model="meta-llama/Llama-3.1-8B",
    prefill=wp.Pool("1xH100", replicas=1),
    decode=wp.Pool("1xH100", replicas=2),
    cloud="lambda",
)
engine.up()
print(engine.status().endpoint)
engine.down()
PY
```

For live Lambda integration, install cloud extras and opt in explicitly:

```bash
pip install -e ".[cloud,dev]"
WARPLY_INTEGRATION=1 pytest tests/test_integration_lambda.py
```

Live integration may launch paid GPU instances.

## Current Limits

- `cloud="local"` is a mock runtime; it does not start SGLang.
- Live cloud `scale()` is not implemented yet; relaunch with a new spec.
- Cloud disagg currently supports `prefill.replicas == 1` and `decode.replicas >= 1`.
- CUDA/SGLang/NIXL is the only live target under active validation.
- AMD Instinct specs such as `wp.Pool("1xMI300X")` compile to ROCm-aware plans, but live ROCm
  launch fails fast until a ROCm image and transfer backend such as MORI are validated.
- KV-aware routing, stats, vLLM/TensorRT-LLM adapters, Dynamo runtime integration, and RL loops
  are roadmap items.

## Architecture

```text
DisaggEngine spec
  -> compiler
  -> DeploymentPlan
  -> provider adapter      SkyPilot, local mock, future direct providers
  -> engine adapter        SGLang now; vLLM / TensorRT-LLM later
  -> KV adapter            NIXL now; MORI / Mooncake / LMCache candidates later
  -> router + client       OpenAI-compatible endpoint
```

Warply is intentionally Python-first. Hot-path serving remains inside engines and runtimes that
already specialize in kernels, batching, scheduling, and transport.

## Roadmap

| Phase | Focus |
| --- | --- |
| Phase 0 | Validate live SGLang/NIXL Lambda serving, add `engine.stats()`, improve 1:N P/D scaling |
| Phase 1 | vLLM adapter, Dynamo runtime target, KV-aware routing, AWS/CoreWeave polish |
| Phase 2 | RL rollout pools, eval/judge pools, self-improvement workflows, policy-driven scaling |
| Later | ROCm live launch, TensorRT-LLM adapter, richer observability, managed control plane |

Track planned work in [GitHub issues](https://github.com/afifi-yusuf/warply/issues).

## Development

```bash
pip install -e ".[dev]"
ruff check warply tests
pytest -q
```

CI runs the same checks on Python 3.10, 3.11, and 3.12. GPU/cloud tests are skipped unless
explicitly enabled with `WARPLY_INTEGRATION=1`.

## Community

- Website: [warply.ai](https://warply.ai)
- Issues: [bugs, feature requests, and design discussions](https://github.com/afifi-yusuf/warply/issues)
- Contributing guide: [CONTRIBUTING.md](./CONTRIBUTING.md)
- Security policy: [SECURITY.md](./SECURITY.md)
- Code of conduct: [CODE_OF_CONDUCT.md](./CODE_OF_CONDUCT.md)

## License

Apache 2.0. See [LICENSE](./LICENSE).

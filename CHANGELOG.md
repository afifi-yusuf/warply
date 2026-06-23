# Changelog

All notable changes to Warply will be documented in this file.

Warply is pre-1.0; public APIs may still change as the deployment model is validated.

## Unreleased

### Added

- Python SDK surface for `DisaggEngine` and `Pool`.
- Deterministic deployment compiler and YAML export.
- Local mock runtime for no-GPU lifecycle testing.
- SGLang process config rendering for prefill, decode, and router roles.
- NIXL transfer config for CUDA/SGLang plans.
- SkyPilot task rendering and dry-run provider path for Lambda-style cloud launches.
- OpenAI-compatible HTTP client wrapper for deployed router endpoints.
- CUDA and ROCm accelerator planning metadata.
- GitHub Actions CI for Python 3.10, 3.11, and 3.12.

### Known Limits

- Live GPU integration is experimental and gated behind `WARPLY_INTEGRATION=1`.
- Cloud `scale()` after `up()` is not implemented yet.
- ROCm plans compile/export, but live AMD launch is not enabled.

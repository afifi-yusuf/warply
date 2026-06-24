# Cloud Providers

Warply uses SkyPilot as its first provisioning backend. SkyPilot gives Warply reach across many
clouds, but Warply support is still validated provider by provider.

## Current Warply Status

| Provider | Warply status | Notes |
| --- | --- | --- |
| Local | Supported mock | No GPU required; exercises lifecycle and client behavior locally. |
| Lambda | Active target | SkyPilot task rendering and dry-run are implemented; live GPU E2E is gated. |
| CoreWeave | Planned | Best reached through SkyPilot Kubernetes support once Lambda is validated. |
| AWS | Planned | Natural next general cloud target after Lambda validation. |
| Modal | Possible future provider | Useful for managed endpoint smoke tests, but not the primary P/D cluster path. |
| AMD/ROCm clouds | Planning only | AMD GPU specs compile/export, but live ROCm launch is not enabled yet. |

## SkyPilot Reach

SkyPilot can provision many backends, including major clouds, GPU neoclouds, Kubernetes, and
Slurm clusters. That does not mean Warply has validated all of them.

As of the current SkyPilot docs, relevant targets include:

- AWS
- GCP
- Azure
- OCI
- Lambda
- Nebius
- RunPod
- Paperspace
- Fluidstack
- Cudo
- Shadeform
- IBM
- SCP
- Seeweb
- vSphere
- Vast
- Kubernetes
- Slurm

## What Warply Needs From A Provider

Disaggregated prefill/decode serving has stricter requirements than a single GPU endpoint:

- Multi-node GPU placement in one deployment or cluster.
- Stable internal addresses for prefill, decode, and router wiring.
- Fast node-to-node networking for KV transfer.
- A public router endpoint for OpenAI-compatible traffic.
- Reliable health checks and teardown.
- Clear support for required engine images, CUDA/ROCm runtime, and transfer backend packages.

## Provider Roadmap

The current default path is:

1. Validate Lambda live SGLang/NIXL serving.
2. Add AWS as the next broad cloud target.
3. Add CoreWeave through Kubernetes once the cluster shape is proven.
4. Evaluate additional neoclouds based on networking, GPU availability, and user demand.
5. Add Modal as a separate managed-endpoint provider if it helps real users test quickly.

Warply should only claim a provider as supported after dry-run tests, task rendering tests, and at
least one documented live integration path exist.

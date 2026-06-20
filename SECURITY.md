# Security Policy

Warply is early-stage infrastructure software for launching and routing inference workloads.
Please report security issues privately so maintainers have time to investigate and prepare a
fix before details are public.

## Supported Versions

Warply is currently pre-1.0. Security fixes are provided for the `main` branch and the latest
published package version when applicable.

| Version | Supported |
| --- | --- |
| `main` | Yes |
| Latest release | Yes |
| Older releases | Best effort |

## Reporting a Vulnerability

Do not open a public GitHub issue for suspected vulnerabilities.

Please report security issues through GitHub Private Vulnerability Reporting:

https://github.com/afifi-yusuf/warply/security/advisories/new

If private vulnerability reporting is unavailable, open a public issue that asks for a
maintainer security contact without disclosing vulnerability details.

Include as much of the following as you can:

- A short description of the issue and its impact.
- Steps to reproduce, proof-of-concept code, or relevant logs.
- Affected versions, commit SHAs, or deployment settings.
- Whether credentials, cloud resources, model endpoints, or user data may be exposed.
- Any suggested mitigations or patches.

You should receive an initial response within 7 days. If the issue is confirmed, maintainers
will coordinate a fix and public disclosure timeline with you.

## Scope

Security-relevant reports include, but are not limited to:

- Credential, token, or secret exposure.
- Unsafe command execution, provisioning, or teardown behavior.
- Cross-tenant or unintended network access in provider integrations.
- Unauthorized access to inference endpoints, routers, or health APIs.
- Supply-chain or dependency issues that affect Warply users.

General bugs, feature requests, and design discussions should use GitHub issues instead.

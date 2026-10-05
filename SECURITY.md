# Security and data handling

## Scope

This repository is a portfolio / demonstration project. Public files and Git history were scanned with Gitleaks 8.30.1, and runtime dependency versions were checked against npm / PyPI vulnerability databases during the October 4, 2026 release. These checks find known patterns and advisories; they are not a penetration test or a guarantee that the application is secure.

Keep secrets in private environment variables or a secret manager. `.env.example` is a blank template. Do not commit populated workflow exports, databases, uploads, customer records, transcripts, recordings or provider keys. A public frontend variable is visible to every visitor.

## Reporting

Report a suspected issue privately through the contact channel at [synqlogic.com](https://synqlogic.com). Include the repository, affected component and a minimal synthetic reproduction. Do not put credentials or personal data in a public GitHub issue. Rotate any exposed live secret immediately; removing it from a file does not invalidate it or erase history.

## Automated controls

The security workflow scans Git history on pushes / pull requests with fully redacted output. Workflow permissions are read-only, official checkout is pinned to a commit, and the scanner archive is verified by SHA-256 before execution. Dependency update checks are configured weekly. Review and test updates before merging them.

## Project-specific boundaries

- Model generation runs locally without external AI keys. It produces plausible text, not reliable facts; do not use it for consequential decisions.
- Checkpoints contain Python objects. The code uses `torch.load(..., weights_only=False)` for this project's custom tokenizer / configuration. Loading an untrusted checkpoint can execute arbitrary code. Only load files you created or trust; never expose an upload-and-load endpoint.
- `models/SHA256SUMS` records the included model's digest. A matching checksum checks integrity against this repository; it does not certify that an arbitrary model is safe.
- The web API limits prompt length, token count, sampling settings and simultaneous generation; it uses security headers and per-process request limits. Add shared quotas for multiple server instances.
- The IP guard trusts forwarded headers and assumes a hosting proxy that overwrites client-supplied values. Configure that boundary if hosting elsewhere.
- No private keys, customer data or general-purpose model file uploads are required by the demo.

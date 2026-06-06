# Secret Risk Audit

## Scope

Scanned parent-repo files in:

- `src/`
- `scripts/`
- `configs/`
- `tests/`
- `docs/`
- `manifests/`
- `patches/`
- `.github/`
- `.env.example`
- `pyproject.toml`
- `requirements/`

Excluded `third_party/` submodule internals and binary-like files.

## Findings

| Path | Risk strings | Classification | Recommended action |
| --- | --- | --- | --- |
| `.env.example` | API key environment variable names | placeholder only | Keep. Values are empty placeholders. |
| `patches/audio-maestro-main/tracked_source_only.patch` | Gemini API key parameter/env-var references | variable name only | Human review before public release; no literal value was printed or identified in this audit. |
| `patches/audio-maestro-main/tracked_changes.patch` | Gemini API key parameter/env-var references | variable name only | Preserve privately; review before public release. |
| `patches/audio-maestro-bak/tracked_changes.patch` | Gemini API key parameter/env-var references | variable name only | Duplicate patch area; review before public release. |
| `patches/desta-grpo/tracked_changes.patch` | `GEMINI_API_KEY` documentation/env-var references | variable name only | Review before public release because it is a mixed workspace patch. |
| `patches/desta-grpo-toolrl/tracked_changes.patch` | generic tokenizer/token strings | normal ML/tokenizer vocabulary | No action for secrets; still review as third-party patch. |
| `docs/extraction_inventory.md`, `docs/extraction_report.md`, `docs/final_safety_report.md`, other audit docs | secret-risk prose, token/password words | audit metadata | Keep private until public docs are reviewed; these are audit records, not live secrets. |
| `configs/`, `src/`, `tests/` | `token`, `max_new_tokens`, tokenizer, environment names | normal ML/tokenizer vocabulary | No secret action required. |

## Possible Live Secrets

No obvious live secret value was identified in this audit. High-risk credential names appear as empty placeholders, environment-variable names, or code parameter names.

## Human Review Required Before Public Release

- Manually inspect patch files that mention Gemini/API-key variables.
- Decide whether audit reports containing old workspace paths and secret-risk notes should remain public.
- Re-run a dedicated secret scanner before changing repository visibility.

## Redaction Note

No secret values are printed in this report. Any line containing a possible key assignment was treated as redaction-sensitive during inspection.

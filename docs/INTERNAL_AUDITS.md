# Internal Audit And Migration Docs

This repository keeps historical audit and migration documents so the cleanup process remains traceable. They are not the recommended starting point for public readers.

## Why These Files Are Kept

- They preserve decisions made while extracting a clean repository from a larger research workspace.
- They record what was copied, skipped, reviewed, patched, or intentionally excluded.
- They help future maintainers understand why datasets, checkpoints, logs, caches, and full working directories are not committed.

## Important Warning

These files may contain historical workspace paths, migration metadata, old repository names, local directory references, or audit-only notes. They should not be treated as setup instructions for new users.

Recommendation: keep them for now, then optionally move them to `docs/internal/` later after a final public-docs review.

## Internal Migration And Audit Files

- [extraction_inventory.md](extraction_inventory.md)
- [extraction_report.md](extraction_report.md)
- [modified_repos_audit.md](modified_repos_audit.md)
- [migration_plan.md](migration_plan.md)
- [final_safety_report.md](final_safety_report.md)
- [gitignore_fix_report.md](gitignore_fix_report.md)
- [patch_status.md](patch_status.md)
- [path_cleanup_report.md](path_cleanup_report.md)
- [remaining_path_cleanup_report.md](remaining_path_cleanup_report.md)
- [local_path_hardening_report.md](local_path_hardening_report.md)
- [post_push_remote_fix.md](post_push_remote_fix.md)
- [github_target.md](github_target.md)
- [repo_status.md](repo_status.md)

## Public-Readiness Audit Files

These are partly historical and partly useful for maintainers. They are safe to keep for now, but should be reviewed before a polished public release.

- [public_readiness_audit.md](public_readiness_audit.md)
- [secret_risk_audit.md](secret_risk_audit.md)
- [public_release_blockers.md](public_release_blockers.md)
- [license_audit.md](license_audit.md)
- [license_notes.md](license_notes.md)
- [remaining_work.md](remaining_work.md)

## Mixed Docs Needing Judgment

- [patch_application_plan.md](patch_application_plan.md)
  Publicly useful for dry-run patch validation, but it also references preservation history.

- [submodule_plan.md](submodule_plan.md)
  Useful for maintainers and technically public-safe, but not essential for most readers.

- [setup_status.md](setup_status.md)
  Useful as a current-status record, but less important than the quickstart and CI docs.

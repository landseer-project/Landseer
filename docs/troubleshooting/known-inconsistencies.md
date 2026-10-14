# Known Inconsistencies

This page tracks current drift between code and older documentation text.

## Current mismatches

- Legacy docs references such as `docs/PIPELINE_RUN_GUIDE.md`, `docs/Tasks.md`, and `docs/OVERVIEWv1.md` appear in comments/readme but are not present as first-class maintained docs.
- Prefer [docs/reference/cli.md](../reference/cli.md): `uv run landseer …`, `uv run landseer-backend|worker|frontend`. Older curl/Poetry examples in long guides may lag.
- Several references still use legacy package paths like `src/landseer_pipeline/*`; active code lives under `src/backend`, `src/worker`, `src/pipeline`, `src/db`, `src/store`, and `src/cli`.

## Policy

When code and docs disagree, trust code first and update docs in the same PR.

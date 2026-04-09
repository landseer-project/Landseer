# Known Inconsistencies

This page tracks current drift between code and older documentation text.

## Current mismatches

- Legacy docs references such as `docs/PIPELINE_RUN_GUIDE.md`, `docs/Tasks.md`, and `docs/OVERVIEWv1.md` appear in comments/readme but are not present as first-class maintained docs.
- Older command examples may reference `poetry run landseer`; current script entry points are `landseer-backend`, `landseer-worker`, and `landseer-frontend`.
- Several references still use legacy package paths like `src/landseer_pipeline/*`; active code lives under `src/backend`, `src/worker`, `src/pipeline`, `src/db`, and `src/store`.

## Policy

When code and docs disagree, trust code first and update docs in the same PR.

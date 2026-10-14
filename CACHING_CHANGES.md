# Caching fixes (branch caching-fixes)

Path: /share/landseer/workspace-ayushi/Landseer-caching (git worktree of ../Landseer, branch `caching-fixes`). Nothing pushed. The live tree was not modified.

## Commits
- f2723ca Snapshot of the simurgh working tree before the caching fixes (HEAD f6f8451 + uncommitted diff + untracked code; left out: *.db files, Landseer.pdf, the paper submodule's dirty contents)
- 042f584 Make the artifact cache safe across hosts and datasets (17 files, +1211 -868)

## What changed
- Store: a `.landseer_complete.json` marker is uploaded last. `artifact_exists` requires the marker or a legacy manifest.json. Downloads and store_local go through a `.tmp-*` staging dir under a per-entry flock, are checked against the marker, then `os.replace`d into place.
- `src/common/content_hash.py`: `dataset_content_hash()` (sha256 over the 4 .npy files, with a size/mtime sidecar `.landseer_hashes.json`).
- Worker cache key: only the dataset content hash + name/variant and the model-script sha256. No paths, MinIO keys or mtimes.
- `dataset_sha256` is recorded on executed results and cache hits, exposed on TaskResponse and parsed by TaskInfo.
- Parent check (local and remote deps): a mismatch fails the task with "parent X was produced on dataset A but this worker has B (LANDSEER_DATA_PATH=...)". An unknown parent hash only warns.
- `/dataset` returns `content_sha256` (cached, computed off the event loop). In manual data-path mode the worker exits at startup on mismatch. On a run change it fails that run's tasks instead (headless backends often have no dataset at startup). `LANDSEER_ALLOW_DATASET_MISMATCH=1` overrides both.
- Pruning skips `.`-names and `datasets/`.
- Removed `src/worker/db.py` and `TaskRunner.run_task_with_cache`. Nothing live used them (only re-exports and one test import; docs/MULTI_PIPELINE_RUN_IMPLEMENTATION.md still mentions them).

## Decisions / findings
- Pre-existing bug fixed: `task_to_response` read `cache_key`/`cache_hit` from the top level of the result, but the worker nests them under `result["artifacts"]`. So the remote-dependency path never got a parent cache key (and never used the cache_key output fallback). It now reads both places. Together with the new fingerprint, this changes every cache key: expect cold-cache misses once after deploying.
- `_task_outputs` keeps its 3-tuple shape (tests depend on it). The parent hash lives in a parallel `_task_dataset_sha` dict.
- A tool output literally named manifest.json shares its object key with the cache manifest (that was already the case). Download copies it back into output when the marker lists it.
- Not done: in auto (non-manual) mode a stale `cache_dir/datasets/<name>/<variant>` download is still reused without comparing hashes. Cache keys and parent checks now catch the consequences, but a re-download on mismatch would be a small follow-up.

## Tests (simurgh venv, per file, `timeout -s KILL 300`, 16 GB ulimit; skipped: test_full_ancestry_chaining.py (needs torch), GPU files)
- Baseline (snapshot): 274 passed, 12 failed. test_worker_comprehensive.py first had to be fixed: 3 work-loop tests (claims_and_executes, handles_no_tasks, handles_exceptions) looped forever and grew memory without bound. The fix is in the commit.
- After: 309 passed, the same 12 failed (MinIO not running locally, default timeout 28000 vs 7200 in a test, 3 in artifact_chaining/task_execution). New tests: 6 content_hash, 8 artifact_completion, 21 dataset_lineage.
- tests/backend/api (4 files touching task/dataset responses): 159 passed, 2 failed both before and after.

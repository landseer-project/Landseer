# Data and Storage

## Database

Database module: `src/db/`

- Default DB type is SQLite (`LANDSEER_DB_TYPE=sqlite`).
- MySQL is supported through environment variables (`LANDSEER_DB_*`).
- Models persist pipelines, workflows, tasks, workers, runs, artifacts, and evaluations.

## Artifact storage

Store module: `src/store/`

- Supports local cache and MinIO-backed object storage.
- Worker can use two-level cache (local + MinIO) when enabled.
- Artifact keys are tied to task identity and dependency lineage.

## Dataset flow

- Backend may prepare dataset and expose dataset metadata through `/dataset`.
- Worker fetches dataset metadata and chooses local path or MinIO download path.

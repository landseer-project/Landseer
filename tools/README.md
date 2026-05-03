# `tools/` — vendored Docker + patches

Contents:

- **`xgbod_v2/`** — `pre_xgbod` (Dockerfiles, `patches/`, `LANDSEER_USAGE.md`)
- **`MagNet/`** — `post_magnet` (Dockerfile, `patches/`, `LANDSEER_USAGE.md`)
- **`evals/`** — metric containers → [`evals/README.md`](evals/README.md)

Build (tags must match `configs/tools.yaml` / `configs/evaluators.example.yaml`):

```bash
cd tools/xgbod_v2 && docker build -f Dockerfile-main -t pre_xgbod:artifact .
cd ../MagNet && docker build -f Dockerfile -t post_magnet:artifact .
```
so on.
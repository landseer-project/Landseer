# `tools/evals`

Evaluator Docker contexts (`evaluate.py` + `Dockerfile` per metric). Shared code: `common/model_loader.py`. **Build context:** this directory (`tools/evals/`).

```bash
cd tools/evals
docker build -f clean/Dockerfile           -t evals/clean:artifact .
docker build -f adversarial/Dockerfile    -t evals/adversarial:artifact .
docker build -f backdoor/Dockerfile       -t evals/backdoor:artifact .
docker build -f fairness/Dockerfile      -t evals/fairness:artifact .
docker build -f fingerprinting/Dockerfile -t evals/fingerprinting:artifact .
docker build -f ood/Dockerfile           -t evals/ood:artifact .
docker build -f watermark/Dockerfile      -t evals/watermark:artifact .
```

`cp configs/evaluators.example.yaml configs/evaluators.yaml` — edit `image:` if your tags differ.

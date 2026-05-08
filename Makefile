.PHONY: docs docs-live docs-linkcheck \
	kind-up kind-down k8s-apply k8s-delete k8s-status k8s-logs k8s-mc

docs:
	python -m pip install -r docs/requirements.txt
	sphinx-build -W -b html docs docs/_build/html

docs-live:
	python -m pip install -r docs/requirements.txt
	sphinx-autobuild docs docs/_build/html

docs-linkcheck:
	python -m pip install -r docs/requirements.txt
	sphinx-build -W -b linkcheck docs docs/_build/linkcheck

# ----------------------------------------------------------------------
# Local Kubernetes development (kind cluster + MinIO)
# ----------------------------------------------------------------------
KIND_CLUSTER ?= landseer
K8S_NAMESPACE ?= landseer-workers

kind-up:
	kind create cluster --name $(KIND_CLUSTER) --config deploy/kind/kind-config.yaml

kind-down:
	kind delete cluster --name $(KIND_CLUSTER)

k8s-apply:
	kubectl apply -k deploy/k8s
	kubectl -n $(K8S_NAMESPACE) rollout status deploy/minio --timeout=120s

k8s-delete:
	kubectl delete -k deploy/k8s --ignore-not-found

k8s-status:
	kubectl -n $(K8S_NAMESPACE) get pods,jobs,svc

# Tail logs from the most recent landseer task Pod.
k8s-logs:
	@pod=$$(kubectl -n $(K8S_NAMESPACE) get pod -l app.kubernetes.io/managed-by=landseer-worker --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}'); \
	if [ -z "$$pod" ]; then echo "No landseer task pods found in $(K8S_NAMESPACE)"; exit 1; fi; \
	echo "Tailing logs for pod: $$pod"; \
	kubectl -n $(K8S_NAMESPACE) logs -f $$pod -c tool

# Open the MinIO web console (uses the host port 9001 mapped by kind-config).
k8s-mc:
	@echo "MinIO console: http://localhost:9001 (user: minioadmin, pass: minioadmin)"
	@echo "MinIO API:     http://localhost:9000"

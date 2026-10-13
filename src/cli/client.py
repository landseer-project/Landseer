"""HTTP client for the Landseer CLI."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urljoin

import httpx


class LandseerApiError(Exception):
    """Raised when the backend returns an error response."""

    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LandseerApiClient:
    """Thin httpx wrapper around Landseer backend endpoints."""

    def __init__(
        self,
        backend_url: str = "http://localhost:8000",
        pipeline_key: Optional[str] = None,
        timeout: float = 60.0,
    ):
        self.backend_url = backend_url.rstrip("/") + "/"
        self.pipeline_key = pipeline_key
        self._client = httpx.Client(timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def _headers(self, *, auth: bool = False) -> Dict[str, str]:
        headers: Dict[str, str] = {"Accept": "application/json"}
        if auth and self.pipeline_key:
            headers["X-Pipeline-Key"] = self.pipeline_key
        return headers

    def _url(self, path: str) -> str:
        return urljoin(self.backend_url, path.lstrip("/"))

    def _request(
        self,
        method: str,
        path: str,
        *,
        auth: bool = False,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
    ) -> Any:
        try:
            response = self._client.request(
                method,
                self._url(path),
                headers=self._headers(auth=auth),
                params=params,
                json=json,
            )
        except httpx.HTTPError as e:
            raise LandseerApiError(f"Request failed: {e}") from e

        if response.status_code >= 400:
            detail: Any
            try:
                payload = response.json()
                detail = payload.get("detail", payload)
            except Exception:
                detail = response.text
            raise LandseerApiError(
                f"HTTP {response.status_code}: {detail}",
                status_code=response.status_code,
                body=detail,
            )

        if response.status_code == 204 or not response.content:
            return None
        try:
            return response.json()
        except Exception:
            return response.text

    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def progress(self, run_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"run_id": run_id} if run_id else None
        return self._request("GET", "/progress", params=params)

    def list_workers(self) -> Dict[str, Any]:
        return self._request("GET", "/workers")

    def list_configs(self) -> Dict[str, Any]:
        return self._request("GET", "/api/pipeline-configs")

    def list_runs(self, config_id: Optional[str] = None) -> Dict[str, Any]:
        params = {"config_id": config_id} if config_id else None
        return self._request("GET", "/api/pipeline-runs", params=params)

    def get_run(self, run_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/api/pipeline-runs/{run_id}")

    def start_run(self, config_id: str) -> Dict[str, Any]:
        """Start a run for a discovered pipeline config id."""
        return self._request(
            "POST",
            f"/api/pipeline-configs/{config_id}/runs",
            auth=True,
            json={"use_cache": True},
        )

    def stop_run(self, run_id: str) -> Dict[str, Any]:
        return self._request("POST", f"/api/pipeline-runs/{run_id}/stop")

    def list_tasks(
        self,
        run_id: str,
        *,
        status: Optional[str] = None,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        params: Dict[str, Any] = {"run_id": run_id, "limit": limit}
        if status:
            params["status"] = status
        return self._request("GET", "/tasks", params=params)

    @staticmethod
    def config_id_from_yaml(path: str) -> str:
        """Map a pipeline YAML path to the backend config id (`config_<stem>`)."""
        from pathlib import Path

        p = Path(path)
        if p.suffix.lower() not in {".yaml", ".yml"}:
            raise LandseerApiError(
                f"Expected a pipeline YAML path, got: {path}"
            )
        return f"config_{p.stem}"

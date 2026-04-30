"""Content-addressable artifact cache (experimental).

Directory layout:
  <root>/<node_hash>/
	 output/ ... tool-produced files ...
	 manifest.json
	 .success

Properties:
  - Append-only: on cache hit we never modify existing files (prevents rewrites).
  - Hash stable across runs: depends only on ordered parent hashes + tool identity subset.
  - Tool identity focuses on semantics (name, docker image & command, params) to avoid
	invalidation for irrelevant config noise.

This module is intentionally dependency-light to keep early import safe.
"""

from __future__ import annotations

import json
import hashlib
import threading
from pathlib import Path
from typing import Dict, List, Any
import stat
import os

from landseer_pipeline.utils.files import hash_file
from landseer_pipeline.container_handler.docker import get_image_digest

try:
	from landseer_pipeline.config import ToolConfig  # type: ignore
except Exception:  # pragma: no cover - during partial import graph
	ToolConfig = object  # fallback placeholder


def _stable_json_hash(obj) -> str:
	data = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
	return hashlib.blake2s(data).hexdigest()


class ArtifactCache:
	"""Minimal content-addressable cache.

	Thread-safe via per-hash locks. Callers must ensure they release locks.
	"""

	def __init__(self, root: Path):
		self.root = Path(root)
		self.root.mkdir(parents=True, exist_ok=True)
		self._locks: Dict[str, threading.Lock] = {}
		self._global = threading.Lock()

	# ---------------- Hash builders -----------------
	def dataset_hash(self, meta_file: Path, variant: str) -> str:
		try:
			meta_txt = meta_file.read_text()
		except Exception:
			meta_txt = ""
		return _stable_json_hash({"variant": variant, "meta_txt": meta_txt})

	def _auxiliary_descriptors(self, tool: ToolConfig) -> List[Dict[str, Any]]:  # type: ignore[override]
		desc: List[Dict[str, Any]] = []
		aux_list = getattr(tool, "auxiliary_files", None) or []
		for aux in aux_list:
			local_path = getattr(aux, 'local_path', None)
			if not local_path:
				continue
			try:
				if os.path.isfile(local_path):
					desc.append({"p": local_path, "h": hash_file(local_path)})
				elif os.path.isdir(local_path):
					# Hash each file (shallow) deterministically
					file_entries = []
					for p in sorted(Path(local_path).rglob('*')):
						if p.is_file():
							try:
								file_entries.append((p.relative_to(local_path).as_posix(), hash_file(str(p))))
							except Exception:
								continue
					desc.append({"p": local_path, "dir": file_entries})
			except Exception:
				continue
		return desc

	def tool_identity_hash(self, tool: ToolConfig) -> str:  # type: ignore[override]
		docker_cfg = getattr(tool, "docker", None)
		image = getattr(docker_cfg, "image", "")
		try:
			digest = get_image_digest(image) if image else ""
		except Exception:
			digest = ""
		ident = {
			"name": getattr(tool, "name", "unknown"),
			"docker_image": image,
			"docker_digest": digest,
			"docker_command": getattr(docker_cfg, "command", ""),
			"params": getattr(tool, "params", {}) or {},
			"required_inputs": getattr(tool, "required_inputs", None) or [],
			"aux": self._auxiliary_descriptors(tool),
		}
		return _stable_json_hash(ident)

	def model_hash(self, script_path: str | None, params: Dict[str, Any]) -> str:
		script_h = None
		if script_path and os.path.exists(script_path):
			try:
				script_h = hash_file(script_path)
			except Exception:
				script_h = "missing"
		return _stable_json_hash({"script": script_h or "none", "params": params or {}})

	def node_hash(self, parents: List[str], tool_hash: str) -> str:
		return _stable_json_hash({"parents": parents, "tool": tool_hash})

	# ---------------- Path helpers ------------------
	def path_for(self, h: str) -> Path:
		return self.root / h

	def exists(self, h: str) -> bool:
		import logging
		logger = logging.getLogger(__name__)
		success_path = self.path_for(h) / ".success"
		result = success_path.exists()
		logger.debug(f"[ARTIFACT] {h[:12]}: exists() check -> {result} (path: {success_path})")
		return result

	# ---------------- Locking -----------------------
	def lock(self, h: str):
		with self._global:
			if h not in self._locks:
				self._locks[h] = threading.Lock()
			lk = self._locks[h]
		lk.acquire()
		return lk

	# ---------------- Manifest ----------------------
	def write_success(self, h: str, manifest: dict):
		import logging
		logger = logging.getLogger(__name__)
		node_dir = self.path_for(h)
		success_marker = node_dir / ".success"
		
		# Idempotent: if already marked successful, don't re-write (files are read-only)
		if success_marker.exists():
			logger.debug(f"[ARTIFACT] {h[:12]}: Already marked successful, skipping")
			return
		
		node_dir.mkdir(parents=True, exist_ok=True)
		(node_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
		success_marker.touch()
		# Force filesystem sync to ensure visibility to other threads
		import os
		os.sync()
		logger.debug(f"[ARTIFACT] {h[:12]}: Marked successful, .success created at {success_marker}")
		# Harden: mark files read-only to avoid accidental rewrites
		try:
			for p in node_dir.rglob("*"):
				if p.is_file():
					p.chmod(stat.S_IREAD | stat.S_IRGRP | stat.S_IROTH)
		except Exception as e:
			logger.warning(f"[ARTIFACT] {h[:12]}: Failed to chmod files read-only: {e}")


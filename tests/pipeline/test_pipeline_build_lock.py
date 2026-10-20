"""
Pipeline builds are serialized.

The task dedup registry in src/pipeline/tasks.py is module-global and
create_pipeline_from_config(clear_registry=True) wipes it, so two runs built at
once (start/restart routes build in worker threads) must not interleave.
"""

import threading
import time

from src.pipeline import config_loader


def test_concurrent_builds_do_not_overlap(monkeypatch):
    active = 0
    max_active = 0
    lock = threading.Lock()
    built = []

    def _fake_build(config_path, **kwargs):
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.05)
        with lock:
            active -= 1
        built.append(config_path)
        return config_path

    monkeypatch.setattr(config_loader, "_build_pipeline_from_config", _fake_build)
    threads = [
        threading.Thread(target=config_loader.create_pipeline_from_config, args=(f"cfg_{i}.yaml",))
        for i in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert max_active == 1
    assert sorted(built) == [f"cfg_{i}.yaml" for i in range(4)]


def test_wrapper_keeps_signature_docs_and_result(monkeypatch):
    assert "Create a complete Pipeline instance" in config_loader.create_pipeline_from_config.__doc__
    monkeypatch.setattr(config_loader, "_build_pipeline_from_config", lambda *a, **k: ("built", a, k))
    assert config_loader.create_pipeline_from_config("x.yaml", clear_registry=False) == (
        "built", ("x.yaml",), {"clear_registry": False}
    )

"""
TensorFlow GPU images must really use the GPU and share it.

tensorflow/tensorflow:*-gpu images list the CUDA stub libcuda.so.1
(/etc/ld.so.conf.d/z-cuda-stubs.conf) in the linker cache ahead of the host
driver's, so TensorFlow loads the stub (cuInit error 34) and silently runs on
the CPU. And TensorFlow reserves almost all GPU memory unless memory growth is
on, starving other tasks on the same GPU. Every Dockerfile under containers/
built on such a base must handle both.
"""

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
TF_GPU_BASE = re.compile(r"^FROM\s+tensorflow/tensorflow:\S*gpu\S*", re.MULTILINE)


def _tf_gpu_dockerfiles():
    return sorted(p for p in (REPO_ROOT / "containers").rglob("Dockerfile")
                  if TF_GPU_BASE.search(p.read_text()))


def test_tool_images_are_found():
    names = {p.parent.name for p in _tf_gpu_dockerfiles()}
    assert {"in_freeat", "in_madry"} <= names


@pytest.mark.parametrize("dockerfile", _tf_gpu_dockerfiles(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_removes_cuda_stub_from_linker_cache(dockerfile):
    text = dockerfile.read_text()
    assert re.search(r"rm\s+(-f\s+)?/etc/ld\.so\.conf\.d/z-cuda-stubs\.conf\s*&&\s*ldconfig", text)


@pytest.mark.parametrize("dockerfile", _tf_gpu_dockerfiles(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_enables_gpu_memory_growth(dockerfile):
    assert re.search(r"TF_FORCE_GPU_ALLOW_GROWTH=true", dockerfile.read_text())

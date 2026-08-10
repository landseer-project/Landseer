#!/usr/bin/env python3
"""
Compare dataset outputs produced by:
1) legacy loader code from `landseer-main`
2) current local clean dataset artifacts consumed by `Landseer`

This is an integration/parity test for the "post-dataloader" artifacts that are
passed downstream to tools (`data.npy`, `labels.npy`, etc.).

By default it tests BOTH datasets:
  - cifar10
  - celeba

Examples:
  python helper_scripts/compare_dataloader_outputs.py
  python helper_scripts/compare_dataloader_outputs.py --dataset cifar10
  python helper_scripts/compare_dataloader_outputs.py --keep-temp
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_DATA_ROOT = Path("/data/landseer/landseer_old_data/data")
DEFAULT_LANDSEER_MAIN = Path("/share/landseer/workspace-ayushi/landseer-main")
DEFAULT_TMP_ROOT = Path("/data/landseer/tmp")
DEFAULT_LEGACY_PYTHON = Path("/share/landseer/workspace-ayushi/.shared/envs/landseer-main/bin/python")


@dataclass
class CompareResult:
    ok: bool
    details: List[str]


def _load_module_from_path(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module {module_name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_legacy_loader(
    *,
    legacy_python: Path,
    loader_path: Path,
    output_dir: Path,
    download_dir: Path,
    celeba_mode: bool,
) -> None:
    if not legacy_python.exists():
        raise FileNotFoundError(
            f"Legacy Python not found: {legacy_python}. "
            "Use --legacy-python to point to an env with torch/torchvision/sklearn/PIL."
        )

    if celeba_mode:
        code = (
            "from pathlib import Path\n"
            f"src = Path(r'''{loader_path}''').read_text(encoding='utf-8')\n"
            "src = src.replace(\n"
            "  'from kaggle.api.kaggle_api_extended import KaggleApi',\n"
            "  'class KaggleApi:\\n"
            "    def authenticate(self):\\n"
            "      return None\\n"
            "    def dataset_download_files(self, *args, **kwargs):\\n"
            "      return None'\n"
            ")\n"
            "ns = {}\n"
            f"exec(compile(src, r'''{loader_path}''', 'exec'), ns)\n"
            f"ns['load_dataset'](output_dir=r'''{output_dir}''', download_dir=r'''{download_dir}''')\n"
        )
    else:
        code = (
            "import importlib.util\n"
            f"spec=importlib.util.spec_from_file_location('legacy_loader', r'''{loader_path}''')\n"
            "mod=importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(mod)\n"
            f"mod.load_dataset(output_dir=r'''{output_dir}''', download_dir=r'''{download_dir}''')\n"
        )

    result = subprocess.run(
        [str(legacy_python), "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "Legacy loader execution failed.\n"
            f"stdout:\n{result.stdout}\n\nstderr:\n{result.stderr}"
        )


def _sha256_npy(path: Path) -> str:
    arr = np.load(path, allow_pickle=True)
    h = hashlib.sha256()
    h.update(str(arr.dtype).encode("utf-8"))
    h.update(str(arr.shape).encode("utf-8"))
    # For object arrays (e.g., filename lists), normalize via string payload.
    if arr.dtype == object:
        payload = "\n".join(map(str, arr.tolist())).encode("utf-8")
        h.update(payload)
    else:
        h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _compare_files(expected: Path, actual: Path, files: Iterable[str]) -> CompareResult:
    details: List[str] = []
    ok = True
    for name in files:
        exp = expected / name
        act = actual / name
        if not exp.exists():
            ok = False
            details.append(f"Missing expected file: {exp}")
            continue
        if not act.exists():
            ok = False
            details.append(f"Missing generated file: {act}")
            continue

        exp_arr = np.load(exp, allow_pickle=True)
        act_arr = np.load(act, allow_pickle=True)

        if exp_arr.shape != act_arr.shape:
            ok = False
            details.append(
                f"{name}: shape mismatch expected={exp_arr.shape} actual={act_arr.shape}"
            )
            continue
        if exp_arr.dtype != act_arr.dtype:
            ok = False
            details.append(
                f"{name}: dtype mismatch expected={exp_arr.dtype} actual={act_arr.dtype}"
            )
            continue

        exp_hash = _sha256_npy(exp)
        act_hash = _sha256_npy(act)
        if exp_hash != act_hash:
            ok = False
            details.append(f"{name}: content hash mismatch")
        else:
            details.append(
                f"{name}: OK shape={exp_arr.shape} dtype={exp_arr.dtype} hash={exp_hash[:12]}"
            )

    return CompareResult(ok=ok, details=details)


def _ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def run_cifar10_compare(
    landseer_main_root: Path,
    data_root: Path,
    out_dir: Path,
    legacy_python: Path,
) -> CompareResult:
    loader_path = (
        landseer_main_root
        / "src/landseer_pipeline/dataset_handler/loaders/cifar10_loader.py"
    )
    _ensure_exists(loader_path, "CIFAR10 loader")

    current_clean = data_root / "cifar10/clean"
    source_download = data_root / "cifar10/downloaded_dataset"
    _ensure_exists(current_clean, "Current Landseer CIFAR-10 clean dir")
    _ensure_exists(source_download, "CIFAR-10 source download dir")

    generated = out_dir / "cifar10_legacy_generated"
    generated.mkdir(parents=True, exist_ok=True)
    _run_legacy_loader(
        legacy_python=legacy_python,
        loader_path=loader_path,
        output_dir=generated,
        download_dir=source_download,
        celeba_mode=False,
    )

    files = ["data.npy", "labels.npy", "test_data.npy", "test_labels.npy"]
    return _compare_files(expected=current_clean, actual=generated, files=files)


def run_celeba_compare(
    landseer_main_root: Path,
    data_root: Path,
    out_dir: Path,
    legacy_python: Path,
) -> CompareResult:
    loader_path = (
        landseer_main_root
        / "src/landseer_pipeline/dataset_handler/loaders/celeba_loader.py"
    )
    _ensure_exists(loader_path, "CelebA loader")

    current_clean = data_root / "celeba/clean"
    source_download = data_root / "celeba/downloaded_dataset"
    _ensure_exists(current_clean, "Current Landseer CelebA clean dir")
    _ensure_exists(source_download, "CelebA source download dir")

    generated = out_dir / "celeba_legacy_generated"
    generated.mkdir(parents=True, exist_ok=True)
    _run_legacy_loader(
        legacy_python=legacy_python,
        loader_path=loader_path,
        output_dir=generated,
        download_dir=source_download,
        celeba_mode=True,
    )

    files = [
        "data.npy",
        "labels.npy",
        "test_data.npy",
        "test_labels.npy",
        "filenames.npy",
        "test_filenames.npy",
    ]
    return _compare_files(expected=current_clean, actual=generated, files=files)


def _print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare Landseer dataset dataloader outputs")
    parser.add_argument(
        "--dataset",
        choices=["cifar10", "celeba", "both"],
        default="both",
        help="Dataset to compare (default: both)",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Local dataset root (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--landseer-main-root",
        type=Path,
        default=DEFAULT_LANDSEER_MAIN,
        help=f"Path to legacy landseer-main repo (default: {DEFAULT_LANDSEER_MAIN})",
    )
    parser.add_argument(
        "--tmp-root",
        type=Path,
        default=DEFAULT_TMP_ROOT,
        help=f"Temporary output root (default: {DEFAULT_TMP_ROOT})",
    )
    parser.add_argument(
        "--legacy-python",
        type=Path,
        default=DEFAULT_LEGACY_PYTHON,
        help=f"Python executable used to run landseer-main loaders (default: {DEFAULT_LEGACY_PYTHON})",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated temp outputs for inspection",
    )
    args = parser.parse_args()

    args.tmp_root.mkdir(parents=True, exist_ok=True)

    failed = False

    if args.keep_temp:
        run_root = args.tmp_root / "dataloader_parity_keep"
        if run_root.exists():
            shutil.rmtree(run_root)
        run_root.mkdir(parents=True, exist_ok=True)
        temp_ctx = None
    else:
        temp_ctx = TemporaryDirectory(prefix="dataloader_parity_", dir=str(args.tmp_root))
        run_root = Path(temp_ctx.name)

    try:
        if args.dataset in ("cifar10", "both"):
            _print_section("CIFAR-10 Parity Check")
            result = run_cifar10_compare(
                args.landseer_main_root, args.data_root, run_root, args.legacy_python
            )
            for line in result.details:
                print(line)
            if not result.ok:
                failed = True
                print("CIFAR-10 parity: FAILED")
            else:
                print("CIFAR-10 parity: OK")

        if args.dataset in ("celeba", "both"):
            _print_section("CelebA Parity Check")
            result = run_celeba_compare(
                args.landseer_main_root, args.data_root, run_root, args.legacy_python
            )
            for line in result.details:
                print(line)
            if not result.ok:
                failed = True
                print("CelebA parity: FAILED")
            else:
                print("CelebA parity: OK")

        if failed:
            print("\nOverall result: FAILED")
            print(f"Generated outputs kept at: {run_root}")
            return 1

        print("\nOverall result: OK")
        if args.keep_temp:
            print(f"Generated outputs kept at: {run_root}")
        return 0
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


if __name__ == "__main__":
    sys.exit(main())


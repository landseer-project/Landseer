"""Filesystem helpers for optional hard-link staging.

When ``LANDSEER_HARDLINK_MACRO`` is enabled, regular files on the same
filesystem share inodes instead of duplicating bytes. Cross-device and
permission failures fall back to ``shutil.copy2``.
"""

from __future__ import annotations

import errno
import filecmp
import os
import shutil
from pathlib import Path
from typing import Union

from .pylogger import get_logger

logger = get_logger(__name__)

PathLike = Union[str, Path]


def hardlink_macro_enabled() -> bool:
    """Return True when hardlink mode is explicitly enabled via env."""
    return os.environ.get("LANDSEER_HARDLINK_MACRO", "").lower() in ("1", "true", "yes")


def same_content(src: PathLike, dst: PathLike) -> bool:
    """Return True when two regular files are byte-for-byte identical."""
    src_path = Path(src)
    dst_path = Path(dst)
    if (
        not src_path.exists()
        or not dst_path.exists()
        or not src_path.is_file()
        or not dst_path.is_file()
    ):
        return False
    try:
        return (
            src_path.stat().st_size == dst_path.stat().st_size
            and filecmp.cmp(src_path, dst_path, shallow=False)
        )
    except OSError:
        return False


def link_or_copy(src: PathLike, dst: PathLike) -> None:
    """Copy a file, or hard-link when the hardlink macro is enabled.

    When the destination already exists and has identical content, replace it
    with a hard link to avoid rewriting bytes. ``shutil.copytree`` may pass
    ``str`` paths as ``copy_function`` arguments, so paths are coerced to
    :class:`~pathlib.Path`.
    """
    src_path = Path(src)
    dst_path = Path(dst)
    if not hardlink_macro_enabled():
        shutil.copy2(src_path, dst_path)
        return

    if dst_path.exists() and dst_path.is_file() and same_content(src_path, dst_path):
        try:
            dst_path.unlink(missing_ok=True)
            os.link(src_path, dst_path)
            return
        except OSError as exc:
            if exc.errno in (errno.EXDEV, errno.EPERM, errno.EACCES):
                shutil.copy2(src_path, dst_path)
                return
            raise

    try:
        os.link(src_path, dst_path)
    except OSError as exc:
        if exc.errno in (errno.EXDEV, errno.EPERM, errno.EACCES):
            shutil.copy2(src_path, dst_path)
        else:
            raise


def copytree_with_optional_hardlinks(src: PathLike, dst: PathLike) -> None:
    """Copy a directory tree, optionally hard-linking files when macro is enabled."""
    shutil.copytree(src, dst, copy_function=link_or_copy)


def mirror_tree_link_or_copy(src_dir: PathLike, dst_dir: PathLike) -> None:
    """Replicate *src_dir* under *dst_dir*, hard-linking regular files when possible."""
    src_path = Path(src_dir)
    dst_path = Path(dst_dir)
    if not src_path.is_dir():
        raise ValueError(f"expected directory: {src_path}")
    for dirpath, _dirnames, filenames in os.walk(src_path, followlinks=False):
        rel = Path(dirpath).relative_to(src_path)
        target_dir = dst_path / rel
        target_dir.mkdir(parents=True, exist_ok=True)
        for name in filenames:
            s = Path(dirpath) / name
            d = target_dir / name
            if d.exists() or d.is_symlink():
                d.unlink()
            if s.is_symlink():
                shutil.copy2(s, d, follow_symlinks=False)
            elif s.is_file():
                link_or_copy(s, d)


def link_identical_output_to_input(output_dir: PathLike, input_dir: PathLike) -> None:
    """Replace output files with hard links to matching input files when content matches."""
    if not hardlink_macro_enabled():
        return

    output_path_root = Path(output_dir)
    input_path_root = Path(input_dir)
    if (
        not output_path_root.exists()
        or not output_path_root.is_dir()
        or not input_path_root.exists()
        or not input_path_root.is_dir()
    ):
        return

    for output_path in output_path_root.rglob("*"):
        if not output_path.is_file() or output_path.is_symlink():
            continue
        rel_path = output_path.relative_to(output_path_root)
        candidate = input_path_root / rel_path
        if not candidate.exists() or not candidate.is_file() or candidate.is_symlink():
            continue
        if not same_content(candidate, output_path):
            continue

        try:
            output_path.unlink(missing_ok=True)
            os.link(candidate, output_path)
            logger.debug(
                "Hard-linked output file to matching input file: %s -> %s",
                output_path,
                candidate,
            )
        except OSError as exc:
            if exc.errno in (errno.EXDEV, errno.EPERM, errno.EACCES):
                logger.debug(
                    "Falling back to regular file for hardlink target due to "
                    "filesystem limitation: %s",
                    output_path,
                )
            else:
                raise

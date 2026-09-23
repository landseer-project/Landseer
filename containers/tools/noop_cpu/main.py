#!/usr/bin/env python3
"""A noop tool that just passes its input through to its output.

We need this for the pre_training, post_training and deployment stages. The
real ones (pre_noop, post_noop, deploy_noop) live on ghcr and that registry is
private, so we can't pull them. A baseline stage isn't supposed to change
anything anyway, so copying the files across is the whole job.

The worker mounts the input at both /input and /data, and expects output in
/output, so we look in either place.
"""

import os
import shutil
from pathlib import Path


def pick_input_dir() -> Path:
    # worker sets INPUT_DIR, but fall back to the usual mount points
    for candidate in (os.environ.get("INPUT_DIR"), "/input", "/data"):
        if candidate and Path(candidate).is_dir():
            return Path(candidate)
    raise FileNotFoundError("no input directory found (looked at INPUT_DIR, /input, /data)")


def main() -> None:
    in_dir = pick_input_dir()
    out_dir = Path(os.environ.get("OUTPUT_DIR", "/output"))
    out_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for item in sorted(in_dir.iterdir()):
        # skip the download cache, it's big and nothing downstream reads it
        if item.name.startswith("_") or item.name.startswith("."):
            continue
        target = out_dir / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)
        copied.append(item.name)

    print(f"[noop_cpu] passed through {len(copied)} items from {in_dir} to {out_dir}")
    for name in copied:
        print(f"[noop_cpu]   {name}")


if __name__ == "__main__":
    main()

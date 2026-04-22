"""Allow `python -m src.worker` without runpy double-import warnings."""

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())

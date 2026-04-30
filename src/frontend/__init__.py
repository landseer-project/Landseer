"""Frontend module for Landseer.

This module contains a React + TypeScript frontend built with Vite.
See docs/FRONTEND.md for documentation.

To start the frontend:
    cd src/frontend
    npm install
    npm run dev

Or use the CLI:
    python -m src.frontend.cli dev
"""

__version__ = "0.1.0"

from pathlib import Path

# Frontend directory path
FRONTEND_DIR = Path(__file__).parent

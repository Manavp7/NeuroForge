"""Dump the FastAPI OpenAPI schema to JSON (offline, no server needed).

Usage:
    python scripts/export_openapi.py [output_path]
Default output: ../openapi.json (repo root).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from neuroforge.api.app import app


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    out = Path(argv[0]) if argv else Path(__file__).resolve().parents[2] / "openapi.json"
    out.write_text(json.dumps(app.openapi(), indent=2))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

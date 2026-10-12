"""Small command-line helper shared by the Stage 3 entry scripts."""

from __future__ import annotations

import sys
from typing import Any


def inject_defaults(defaults: dict[str, Any]) -> None:
    """Add Stage 3 defaults while preserving explicit command-line overrides."""
    present = {
        item.split("=", 1)[0]
        for item in sys.argv[1:]
        if str(item).startswith("--")
    }
    additions: list[str] = []
    for key, value in defaults.items():
        flag = f"--{key}"
        if flag in present:
            continue
        if isinstance(value, bool):
            if value:
                additions.append(flag)
            continue
        additions.extend([flag, str(value)])
    sys.argv = [sys.argv[0], *additions, *sys.argv[1:]]

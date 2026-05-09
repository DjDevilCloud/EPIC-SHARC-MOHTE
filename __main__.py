"""Module entry point for EPIC-SHARC MOHTE."""

# SPDX-License-Identifier: AGPL-3.0-or-later
try:
    from .cli import main
except ImportError:  # pragma: no cover - supports direct script launching.
    from cli import main


if __name__ == "__main__":
    raise SystemExit(main())

"""`ase` command-line entry point (subcommand dispatch only)."""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    del args  # subcommands wired in later steps
    print("agent-skill-eval: CLI not wired yet")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

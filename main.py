"""Entry point used by PyInstaller to build agent.exe (keeps package imports working)."""

import sys

from agent.cli import main

if __name__ == "__main__":
    sys.exit(main())

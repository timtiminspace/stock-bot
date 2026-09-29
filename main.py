"""Repository-level convenience entry point for Market Signal Lab."""

import runpy
import sys
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent / "marketsignallab" / "src"


def main() -> None:
    sys.path.insert(0, str(SOURCE_DIR))
    runpy.run_path(str(SOURCE_DIR / "main.py"), run_name="__main__")


if __name__ == "__main__":
    main()

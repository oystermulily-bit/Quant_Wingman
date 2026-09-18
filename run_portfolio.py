"""Start the local offline score / whitelist / allocation workspace."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# Optional project-local dependencies, never a global environment mutation.
runtime = ROOT / ".demo_runtime"
if runtime.is_dir():
    sys.path.insert(0, str(runtime))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    import uvicorn
    print(f"Offline portfolio workspace: http://127.0.0.1:{args.port}/portfolio", flush=True)
    uvicorn.run("web.portfolio_app:app", host="127.0.0.1", port=args.port,
                reload=False, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()

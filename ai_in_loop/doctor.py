from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib import metadata


def _pkg_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def collect_info() -> dict:
    """Collect minimal environment information for debugging."""
    info = {
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "pip_version": _pkg_version("pip"),
        "packages": {
            "langchain": _pkg_version("langchain"),
            "langgraph": _pkg_version("langgraph"),
            "langchain-google-genai": _pkg_version("langchain-google-genai"),
            "langchain-community": _pkg_version("langchain-community"),
            "rank-bm25": _pkg_version("rank-bm25"),
            "google-genai": _pkg_version("google-genai"),
            "python-dotenv": _pkg_version("python-dotenv"),
            "streamlit": _pkg_version("streamlit"),
        },
    }
    return info


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print environment diagnostics for the String Cheese app.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    args = parser.parse_args(argv)

    info = collect_info()
    if args.json:
        print(json.dumps(info, indent=2, sort_keys=True))
    else:
        print("String Cheese Doctor")
        print("====================")
        print(f"Python: {info['python_version']} ({info['python_executable']})")
        print(f"Platform: {info['platform']}")
        print(f"pip: {info['pip_version']}")
        print("Packages:")
        for k, v in info["packages"].items():
            print(f"  - {k}: {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

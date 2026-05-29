#!/usr/bin/env python3
"""Zero-dependency test runner.

Discovers ``tests/test_*.py`` and runs every ``test_*`` function with plain
asserts. Use this when pytest is unavailable; otherwise just run ``pytest``.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    test_files = sorted((ROOT / "tests").glob("test_*.py"))
    passed = failed = 0
    failures: list[str] = []
    for f in test_files:
        mod = _load(f)
        for name in dir(mod):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {f.name}::{name}")
            except Exception:  # noqa: BLE001
                failed += 1
                failures.append(f"{f.name}::{name}\n{traceback.format_exc()}")
                print(f"FAIL {f.name}::{name}")
    print("\n" + "=" * 60)
    print(f"{passed} passed, {failed} failed")
    if failures:
        print("\n--- FAILURES ---")
        for fl in failures:
            print(fl)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

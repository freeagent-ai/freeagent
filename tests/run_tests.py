"""Minimal test runner for machines without pytest: python tests/run_tests.py"""
import importlib.util
import pathlib
import time
import traceback

passed = failed = 0
for path in sorted(pathlib.Path(__file__).parent.glob("test_*.py")):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in sorted(n for n in dir(mod) if n.startswith("test_")):
        t = time.time()
        try:
            getattr(mod, name)()
            passed += 1
            print(f"PASS {path.stem}.{name} ({time.time() - t:.1f}s)")
        except Exception:
            failed += 1
            print(f"FAIL {path.stem}.{name}")
            traceback.print_exc()
print(f"\n{passed} passed, {failed} failed")
raise SystemExit(1 if failed else 0)

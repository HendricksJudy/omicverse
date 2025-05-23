"""Environment checker for OmicVerse.

This module provides utilities to verify that required
packages are installed with compatible versions. It can be
used interactively as a simple wizard after installation.
"""
from __future__ import annotations

import importlib
import sys
from typing import Dict, Tuple

try:
    from packaging.version import Version
except Exception:
    Version = None  # type: ignore


_MIN_DEPENDENCIES = {
    "numpy": "1.23",
    "pandas": "1.5",
    "scanpy": "1.9",
    "torch": "1.13",
}


def _check_package(name: str, min_version: str | None = None) -> Tuple[bool, str]:
    """Check if *name* is importable and meets ``min_version``.

    Parameters
    ----------
    name:
        Package name to check.
    min_version:
        Minimum required version.

    Returns
    -------
    installed:
        Whether the package is importable.
    version:
        Installed version string or ``""`` if not available.
    """
    spec = importlib.util.find_spec(name)
    if spec is None:
        return False, ""
    module = importlib.import_module(name)
    ver = getattr(module, "__version__", "")
    if min_version and Version is not None and ver:
        try:
            if Version(ver) < Version(min_version):
                return False, ver
        except Exception:
            pass
    return True, ver


def check_environment(full: bool = False) -> Dict[str, Tuple[bool, str]]:
    """Verify that key dependencies are installed.

    Parameters
    ----------
    full:
        If ``True``, check additional optional dependencies.

    Returns
    -------
    Dictionary mapping package names to ``(installed, version)``.
    """
    deps = dict(_MIN_DEPENDENCIES)
    if full:
        deps.update({
            "torch_geometric": "2.0",
            "numba": "0.56",
        })

    results: Dict[str, Tuple[bool, str]] = {}
    for name, ver in deps.items():
        results[name] = _check_package(name, ver)
    return results


def environment_wizard() -> None:
    """Interactive wizard to check the current environment."""
    print("OmicVerse environment checker")
    print("1. Quick check")
    print("2. Full check (includes optional GPU libraries)")
    choice = input("Select option [1/2]: ").strip()
    full = choice == "2"
    results = check_environment(full=full)

    py_ver = ".".join(map(str, sys.version_info[:3]))
    print(f"Python version: {py_ver}")
    for pkg, (ok, ver) in results.items():
        status = "OK" if ok else "Missing or incompatible"
        ver_str = ver if ver else "not installed"
        print(f"{pkg}: {ver_str} -> {status}")


if __name__ == "__main__":
    environment_wizard()

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def executor_and_security_config():
    omicverse_pkg = types.ModuleType("omicverse")
    omicverse_pkg.__path__ = [str(PROJECT_ROOT / "omicverse")]
    omicverse_pkg.__file__ = str(PROJECT_ROOT / "omicverse" / "__init__.py")
    sys.modules.setdefault("omicverse", omicverse_pkg)

    # Ensure __file__ is set on the real module (its custom __getattr__
    # raises AttributeError for attributes not in its lazy-load map).
    ov_mod = sys.modules["omicverse"]
    if "__file__" not in vars(ov_mod):
        ov_mod.__file__ = str(PROJECT_ROOT / "omicverse" / "__init__.py")

    utils_pkg = types.ModuleType("omicverse.utils")
    utils_pkg.__path__ = [str(PROJECT_ROOT / "omicverse" / "utils")]
    sys.modules.setdefault("omicverse.utils", utils_pkg)

    ovagent_pkg = types.ModuleType("omicverse.utils.ovagent")
    ovagent_pkg.__path__ = [str(PROJECT_ROOT / "omicverse" / "utils" / "ovagent")]
    sys.modules.setdefault("omicverse.utils.ovagent", ovagent_pkg)

    registry_stub = types.ModuleType("omicverse._registry")
    registry_stub._global_registry = SimpleNamespace(check_prerequisites=lambda *_a, **_k: {"satisfied": True})
    sys.modules.setdefault("omicverse._registry", registry_stub)

    reporter_stub = types.ModuleType("omicverse.utils.agent_reporter")
    reporter_stub.EventLevel = SimpleNamespace(INFO="info")
    sys.modules.setdefault("omicverse.utils.agent_reporter", reporter_stub)

    _load_module("omicverse.utils.agent_errors", PROJECT_ROOT / "omicverse" / "utils" / "agent_errors.py")
    _load_module("omicverse.utils.agent_config", PROJECT_ROOT / "omicverse" / "utils" / "agent_config.py")
    sandbox_module = _load_module("omicverse.utils.agent_sandbox", PROJECT_ROOT / "omicverse" / "utils" / "agent_sandbox.py")
    exec_module = _load_module(
        "omicverse.utils.ovagent.analysis_executor",
        PROJECT_ROOT / "omicverse" / "utils" / "ovagent" / "analysis_executor.py",
    )
    return exec_module.AnalysisExecutor, sandbox_module.SecurityConfig


@pytest.fixture
def sandbox_builtins(executor_and_security_config):
    AnalysisExecutor, SecurityConfig = executor_and_security_config
    ctx = SimpleNamespace(_security_config=SecurityConfig())
    executor = AnalysisExecutor(ctx)
    sandbox_globals = executor.build_sandbox_globals()
    return sandbox_globals["__builtins__"]


def test_denylisted_import_cannot_be_unblocked_by_package_spoofing(sandbox_builtins):
    attacker_globals = {
        "__builtins__": sandbox_builtins,
        "__name__": "omicverse.attacker",
        "__package__": "omicverse",
    }
    payload = compile("import urllib.request", "<agent_generated>", "exec")

    with pytest.raises(ImportError, match="blocked inside the OmicVerse agent sandbox"):
        exec(payload, attacker_globals, {})


def test_omicverse_internal_module_can_still_import_denylisted_module(sandbox_builtins):
    # Use the same __file__ path that the fixture ensured is set on the
    # omicverse module, matching what build_sandbox_globals uses.
    ov_mod = sys.modules["omicverse"]
    ov_root = Path(ov_mod.__file__).resolve().parent
    internal_path = str(ov_root / "biocontext" / "_client.py")

    internal_globals = {
        "__builtins__": sandbox_builtins,
        "__name__": "omicverse.biocontext._client",
        "__package__": "omicverse.biocontext",
    }
    internal_code = compile("import urllib.request", internal_path, "exec")

    exec(internal_code, internal_globals, {})

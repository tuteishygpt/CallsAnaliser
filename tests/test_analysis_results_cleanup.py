from datetime import date
import importlib.util
from types import SimpleNamespace
from pathlib import Path
import subprocess
import sys

from calls_analyser.analysis_results_cleanup import cleanup_analysis_results


class _Query:
    def __init__(self, table, action):
        self.table = table
        self.action = action

    def eq(self, column, value):
        self.table.calls.append((self.action, "eq", column, value))
        return self

    def in_(self, column, values):
        self.table.calls.append((self.action, "in", column, list(values)))
        return self

    def execute(self):
        self.table.calls.append((self.action, "execute"))
        if self.action == "select":
            return SimpleNamespace(data=[{"call_unique_id": "call-1"}])
        return SimpleNamespace(data=[])


class _Table:
    def __init__(self):
        self.calls = []

    def select(self, columns):
        self.calls.append(("select", columns))
        return _Query(self, "select")

    def delete(self):
        self.calls.append(("delete",))
        return _Query(self, "delete")


class _UnexpectedResultTable(_Table):
    def select(self, columns):
        self.calls.append(("select", columns))
        query = _Query(self, "select")

        def execute():
            self.calls.append(("select", "execute"))
            return SimpleNamespace(
                data=[{"call_unique_id": "call-1"}, {"call_unique_id": "unrelated-call"}]
            )

        query.execute = execute
        return query


class _FakeSupabaseCache:
    def __init__(self, table):
        self._table = table


def _cleanup_script_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "clear_yesterday_analysis_results.py"
    spec = importlib.util.spec_from_file_location("cleanup_script_for_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_cleanup_analysis_results_deletes_only_matching_tenant_call_ids_when_executing():
    table = _Table()

    deleted = cleanup_analysis_results(
        table,
        tenant_id="amedis",
        call_unique_ids=["call-1", "call-2"],
        execute=True,
    )

    assert deleted == ["call-1"]
    assert ("select", "eq", "tenant_id", "amedis") in table.calls
    assert ("select", "in", "call_unique_id", ["call-1", "call-2"]) in table.calls
    assert ("delete", "eq", "tenant_id", "amedis") in table.calls
    assert ("delete", "in", "call_unique_id", ["call-1"]) in table.calls


def test_cleanup_analysis_results_dry_run_does_not_delete():
    table = _Table()

    deleted = cleanup_analysis_results(
        table,
        tenant_id="amedis",
        call_unique_ids=["call-1"],
        execute=False,
    )

    assert deleted == ["call-1"]
    assert not any(call[0] == "delete" for call in table.calls)


def test_cleanup_analysis_results_never_deletes_an_id_not_requested():
    table = _UnexpectedResultTable()

    deleted = cleanup_analysis_results(
        table,
        tenant_id="amedis",
        call_unique_ids=["call-1"],
        execute=True,
    )

    assert deleted == ["call-1"]
    assert ("delete", "in", "call_unique_id", ["call-1"]) in table.calls


def test_cleanup_analysis_results_skips_database_when_no_calls_are_selected():
    table = _Table()

    deleted = cleanup_analysis_results(
        table,
        tenant_id="amedis",
        call_unique_ids=[],
        execute=True,
    )

    assert deleted == []
    assert table.calls == []


def test_cleanup_script_uses_fixed_tenant_and_passes_selected_date_to_call_log():
    module = _cleanup_script_module()
    table = _Table()
    tenant = SimpleNamespace(tenant_id="amedis")
    call_log = SimpleNamespace(calls=[])
    call_log.list_calls = lambda selected_day, selected_tenant: (
        call_log.calls.append((selected_day, selected_tenant)) or []
    )
    deps = SimpleNamespace(
        tenant_service=SimpleNamespace(resolve=lambda tenant_id: tenant),
        call_log_service=call_log,
        analysis_service=SimpleNamespace(_cache=_FakeSupabaseCache(table)),
    )
    module.SupabaseCache = _FakeSupabaseCache

    exit_code = module.main(
        args=SimpleNamespace(date=date(2026, 7, 22), execute=False),
        deps=deps,
    )

    assert exit_code == 0
    assert call_log.calls == [(date(2026, 7, 22), tenant)]
    assert table.calls == []


def test_cleanup_script_loads_project_env_before_building_dependencies(monkeypatch):
    module = _cleanup_script_module()
    loaded_paths = []
    monkeypatch.setattr(module, "load_dotenv", lambda path: loaded_paths.append(path))

    module.load_project_env()

    assert loaded_paths == [module.PROJECT_ROOT / ".env"]


def test_cleanup_script_uses_database_tenant_id_casing():
    module = _cleanup_script_module()

    assert module.DEFAULT_TENANT_ID == "Amedis"


def test_cleanup_script_runs_directly_from_project_root():
    project_root = Path(__file__).resolve().parents[1]

    result = subprocess.run(
        [sys.executable, "scripts/clear_yesterday_analysis_results.py", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "--execute" in result.stdout
    assert "--tenant-id" not in result.stdout

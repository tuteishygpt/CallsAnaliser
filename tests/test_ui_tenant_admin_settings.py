from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from calls_analyser.services.auth import TenantSummary
from calls_analyser.services.tenant_admin_settings import TenantAdminValidationError
from calls_analyser.ui.handlers import UIHandlers


class _Auth:
    def __init__(self, allowed=True, tenants=None):
        self.allowed = allowed
        self.checks = []
        self.tenants = tenants or [TenantSummary("inactive", "Inactive tenant", "admin")]

    def authenticate(self, *_args):
        return None

    def list_admin_tenants(self, user_id, include_inactive=True):
        assert include_inactive is True
        return self.tenants

    def can_administer_tenant(self, user_id, tenant_id):
        self.checks.append((user_id, tenant_id))
        return self.allowed


class _AdminService:
    def __init__(self):
        self.calls = []
        self.error = None
        self.saved_document = None

    def load(self, tenant_id):
        self.calls.append(("load", tenant_id))
        return {"tenant_id": tenant_id, "display_name": "A"}

    def save(self, tenant_id, document, actor):
        self.calls.append(("save", tenant_id, actor, document))
        if self.error:
            raise self.error
        return self.saved_document or {**document, "tenant_id": tenant_id}


def _handlers(allowed=True, tenants=None):
    auth = _Auth(allowed, tenants)
    admin = _AdminService()
    handlers = UIHandlers(SimpleNamespace(auth_service=auth, tenant_admin_settings_service=admin))
    return handlers, auth, admin


def test_refresh_uses_live_admin_list_and_includes_inactive_tenant() -> None:
    handlers, _, _ = _handlers()
    dropdown, tab = handlers.refresh_admin_tenants(
        {"authenticated": True, "user_id": "user-1"}
    )
    assert dropdown["choices"] == [("Inactive tenant (admin)", "inactive")]
    assert dropdown["value"] == "inactive"
    assert tab["visible"] is True


def test_refresh_preserves_current_live_selection_and_clears_stale_selection() -> None:
    tenants = [
        TenantSummary("one", "One", "admin"),
        TenantSummary("two", "Two", "admin"),
    ]
    handlers, auth, _ = _handlers(tenants=tenants)
    session = {"authenticated": True, "user_id": "user-1"}

    dropdown, tab = handlers.refresh_admin_tenants(session, "two")
    assert dropdown["value"] == "two"
    assert tab["visible"] is True

    auth.tenants = []
    dropdown, tab = handlers.refresh_admin_tenants(session, "two")
    assert dropdown["choices"] == []
    assert dropdown["value"] is None
    assert tab["visible"] is False


def test_every_load_and_save_uses_live_check_before_admin_service() -> None:
    handlers, auth, admin = _handlers()
    session = {"authenticated": True, "user_id": "user-1"}
    loaded, load_status = handlers.load_tenant_admin_settings("inactive", session)
    saved, save_status = handlers.save_tenant_admin_settings(
        "inactive", {"display_name": "Updated"}, session
    )
    assert loaded["tenant_id"] == "inactive"
    assert saved["display_name"] == "Updated"
    assert auth.checks == [("user-1", "inactive"), ("user-1", "inactive")]
    assert admin.calls[0] == ("load", "inactive")
    assert admin.calls[1][:3] == ("save", "inactive", "user-1")
    assert "secret" not in (load_status + save_status).casefold()


def test_denied_or_forged_tenant_makes_zero_admin_service_calls() -> None:
    handlers, auth, admin = _handlers(False)
    session = {"authenticated": True, "user_id": "user-1"}
    assert handlers.load_tenant_admin_settings("forged", session)[0] == {}
    assert handlers.save_tenant_admin_settings("forged", {"secret": "never"}, session)[0] == {}
    assert auth.checks == [("user-1", "forged"), ("user-1", "forged")]
    assert admin.calls == []


def test_only_dedicated_validation_errors_are_shown_to_the_user() -> None:
    handlers, _, admin = _handlers()
    session = {"authenticated": True, "user_id": "user-1"}
    admin.error = TenantAdminValidationError("Invalid value: General / display name")
    assert handlers.save_tenant_admin_settings("inactive", {}, session)[1] == (
        "Invalid value: General / display name"
    )

    admin.error = ValueError("enc:v1:submitted-secret")
    assert handlers.save_tenant_admin_settings("inactive", {}, session)[1] == (
        "Unable to save tenant settings."
    )


def test_empty_selection_returns_disabled_empty_form() -> None:
    handlers, auth, admin = _handlers()
    result = handlers.load_tenant_admin_form(None, {"authenticated": True, "user_id": "user-1"})

    assert len(result) == len(UIHandlers.TENANT_ADMIN_OUTPUT_FIELDS) + 3
    assert result[0]["value"] is None
    assert result[0]["interactive"] is False
    assert all(item["interactive"] is False for item in result[1:-3])
    assert result[-4]["value"] == []
    assert result[-2]["interactive"] is False
    assert result[-1]["interactive"] is False
    assert auth.checks == []
    assert admin.calls == []


def test_loaded_form_has_read_only_prompts_and_no_ui_defaults() -> None:
    handlers, _, admin = _handlers()
    admin.load = lambda tenant_id: {
        "tenant_id": tenant_id,
        "display_name": "Tenant",
        "status": "inactive",
        "batch_size": "",
        "scheduler_mode": "",
        "scheduler_cron_time": "",
        "scheduler_interval_minutes": "",
        "active_prompts": [
            {"key": "simple", "title": "Simple", "body": "Read only", "version": 3}
        ],
    }

    result = handlers.load_tenant_admin_form(
        "inactive", {"authenticated": True, "user_id": "user-1"}
    )
    by_field = dict(zip(UIHandlers.TENANT_ADMIN_OUTPUT_FIELDS, result))

    assert by_field["batch_size"]["value"] == ""
    assert by_field["scheduler_mode"]["value"] == ""
    assert by_field["scheduler_cron_time"]["value"] == ""
    assert by_field["scheduler_interval_minutes"]["value"] == ""
    assert by_field["active_prompts"]["value"] == [
        ["simple", "Simple", "Read only", 3]
    ]
    assert by_field["active_prompts"].get("interactive") is False
    assert not {
        "prompt_key", "prompt_title", "prompt_body", "prompt_active",
        "prompt_history", "additional_settings", "additional_secrets",
    } & set(UIHandlers.TENANT_ADMIN_OUTPUT_FIELDS)


def test_save_form_sends_only_explicit_editable_fields_and_returns_persisted_readback() -> None:
    handlers, _, admin = _handlers()
    admin.saved_document = {
        "tenant_id": "inactive",
        "display_name": "Persisted",
        "status": "active",
        "active_prompts": [],
    }
    values = [
        "Updated" if field == "display_name" else "active" if field == "status" else ""
        for field in UIHandlers.TENANT_ADMIN_EDITABLE_FIELDS
    ]

    result = handlers.save_tenant_admin_form(
        "inactive", *values, {"authenticated": True, "user_id": "user-1"}
    )
    saved = admin.calls[-1][3]
    by_field = dict(zip(UIHandlers.TENANT_ADMIN_OUTPUT_FIELDS, result))

    assert set(saved) == {"tenant_id", *UIHandlers.TENANT_ADMIN_EDITABLE_FIELDS}
    assert by_field["display_name"]["value"] == "Persisted"
    assert result[-3] == "Settings saved."


def test_layout_refreshes_live_admin_state_after_each_admin_operation() -> None:
    source = Path("calls_analyser/ui/layout.py").read_text(encoding="utf-8")

    assert 'with gr.Tab("Tenant Settings", visible=False)' in source
    assert "login_event.then(" in source
    assert source.count("handlers.refresh_admin_tenants,") == 4
    assert "tenant_admin_change_event = tenant_admin_dd.input(" in source
    assert "tenant_admin_reload_event = tenant_admin_reload_btn.click(" in source
    assert "tenant_admin_save_event = tenant_admin_save_btn.click(" in source
    change_chain = source.split("tenant_admin_change_event =", 1)[1].split(
        "tenant_admin_reload_event =", 1
    )[0]
    reload_chain = source.split("tenant_admin_reload_event =", 1)[1].split(
        "tenant_admin_save_event =", 1
    )[0]
    save_chain = source.split("tenant_admin_save_event =", 1)[1].split(
        "\n\n        filter_btn.click", 1
    )[0]

    for chain in (change_chain, reload_chain):
        assert chain.count("handlers.refresh_admin_tenants") == 1
        assert chain.count("handlers.load_tenant_admin_form") == 1
        assert chain.index("handlers.refresh_admin_tenants") < chain.index(
            "handlers.load_tenant_admin_form"
        )
        assert "inputs=[auth_session, tenant_admin_dd]" in chain
        assert "outputs=[tenant_admin_dd, tenant_admin_tab]" in chain
        assert "inputs=[tenant_admin_dd, auth_session]" in chain
        assert "outputs=tenant_admin_outputs" in chain

    assert save_chain.count("handlers.save_tenant_admin_form") == 1
    assert save_chain.count("handlers.refresh_admin_tenants") == 1
    assert "handlers.load_tenant_admin_form" not in save_chain
    assert save_chain.index("handlers.save_tenant_admin_form") < save_chain.index(
        "handlers.refresh_admin_tenants"
    )
    save_registration, save_refresh = save_chain.split(
        "tenant_admin_save_event.then(", 1
    )
    assert "outputs=tenant_admin_outputs" in save_registration
    assert "inputs=[auth_session, tenant_admin_dd]" in save_refresh
    assert "outputs=[tenant_admin_dd, tenant_admin_tab]" in save_refresh


def test_layout_save_inputs_exclude_read_only_outputs_and_keep_runtime_selectors() -> None:
    source = Path("calls_analyser/ui/layout.py").read_text(encoding="utf-8")
    save_inputs = source.split("tenant_admin_inputs = [", 1)[1].split("\n        ]", 1)[0]

    assert "tenant_admin_prompts" not in save_inputs
    assert "tenant_admin_status," not in save_inputs
    assert "tenant_admin_id" not in save_inputs
    assert "tenant_admin_name" in save_inputs
    assert "tenant_admin_email_name" in save_inputs
    assert "inputs=[tenant_admin_dd, *tenant_admin_inputs, auth_session]" in source
    assert "tenant_selector = tenant_dd if auth_mode else tenant_tb" in source
    assert "tenant_report_selector = tenant_report_dd if auth_mode else tenant_report_tb" in source

from __future__ import annotations

from calls_analyser.ui import dependencies


class FakeEmailReportService:
    def __init__(self, mail_port, *, sender: str, recipient: str) -> None:  # noqa: ANN001
        self.mail_port = mail_port
        self.sender = sender
        self.recipient = recipient


class FakeBrevoAdapter:
    created = 0
    sender_names: list[str | None] = []

    @classmethod
    def from_env(cls, *, sender_name: str | None = None):
        cls.created += 1
        cls.sender_names.append(sender_name)
        return "brevo-port"


class FakeGmailAdapter:
    created = 0

    @classmethod
    def from_env(cls):
        cls.created += 1
        return "gmail-port"


def test_email_report_service_prefers_brevo_when_api_key_is_configured(monkeypatch) -> None:
    FakeBrevoAdapter.created = 0
    FakeGmailAdapter.created = 0
    monkeypatch.setenv("BREVO_API_KEY", "brevo-secret")
    monkeypatch.setenv("GOOGLE_app", "gmail-secret")
    monkeypatch.setenv("EMAIL_FROM", "reports@example.com")
    monkeypatch.setenv("EMAIL_TO", "owner@example.com")
    monkeypatch.setattr(dependencies, "EmailReportService", FakeEmailReportService)
    monkeypatch.setattr(dependencies, "BrevoHTTPSAdapter", FakeBrevoAdapter)
    monkeypatch.setattr(dependencies, "GmailSMTPAdapter", FakeGmailAdapter)

    service = dependencies._build_email_report_service()

    assert service.mail_port == "brevo-port"
    assert service.sender == "reports@example.com"
    assert service.recipient == "owner@example.com"
    assert FakeBrevoAdapter.created == 1
    assert FakeGmailAdapter.created == 0


def test_email_report_service_falls_back_to_gmail_without_brevo_key(monkeypatch) -> None:
    FakeBrevoAdapter.created = 0
    FakeGmailAdapter.created = 0
    monkeypatch.delenv("BREVO_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_app", "gmail-secret")
    monkeypatch.delenv("EMAIL_FROM", raising=False)
    monkeypatch.delenv("EMAIL_TO", raising=False)
    monkeypatch.setattr(dependencies, "EmailReportService", FakeEmailReportService)
    monkeypatch.setattr(dependencies, "BrevoHTTPSAdapter", FakeBrevoAdapter)
    monkeypatch.setattr(dependencies, "GmailSMTPAdapter", FakeGmailAdapter)

    service = dependencies._build_email_report_service()

    assert service.mail_port == "gmail-port"
    assert service.sender == "tuttstt@gmail.com"
    assert service.recipient == "tuttstt@gmail.com"
    assert FakeBrevoAdapter.created == 0
    assert FakeGmailAdapter.created == 1


def test_tenant_email_settings_override_environment_values(monkeypatch) -> None:
    class TenantSettingsService:
        @staticmethod
        def resolve(tenant_id: str):
            assert tenant_id == "amedis"
            return type(
                "TenantSettings",
                (),
                {
                    "email_to": "recipient@amedis.example",
                    "email_from": "sender@amedis.example",
                    "email_from_name": "Amedis calls",
                },
            )()

    FakeBrevoAdapter.created = 0
    FakeBrevoAdapter.sender_names = []
    monkeypatch.setenv("BREVO_API_KEY", "brevo-secret")
    monkeypatch.setenv("EMAIL_FROM", "global-sender@example.com")
    monkeypatch.setenv("EMAIL_TO", "global-recipient@example.com")
    monkeypatch.setattr(dependencies, "EmailReportService", FakeEmailReportService)
    monkeypatch.setattr(dependencies, "BrevoHTTPSAdapter", FakeBrevoAdapter)

    service = dependencies.build_email_report_service_for_tenant(
        TenantSettingsService(),
        "amedis",
    )

    assert service.sender == "sender@amedis.example"
    assert service.recipient == "recipient@amedis.example"
    assert FakeBrevoAdapter.sender_names == ["Amedis calls"]

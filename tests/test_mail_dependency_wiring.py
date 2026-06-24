from __future__ import annotations

from calls_analyser.ui import dependencies


class FakeEmailReportService:
    def __init__(self, mail_port, *, sender: str, recipient: str) -> None:  # noqa: ANN001
        self.mail_port = mail_port
        self.sender = sender
        self.recipient = recipient


class FakeBrevoAdapter:
    created = 0

    @classmethod
    def from_env(cls):
        cls.created += 1
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

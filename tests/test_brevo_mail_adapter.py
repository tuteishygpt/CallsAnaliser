from __future__ import annotations

import base64

from calls_analyser.adapters.mail.brevo import BrevoHTTPSAdapter
from calls_analyser.ports.mail import MailMessage


class FakeResponse:
    def __init__(self) -> None:
        self.raise_for_status_called = False

    def raise_for_status(self) -> None:
        self.raise_for_status_called = True


class FakeHTTPPost:
    def __init__(self) -> None:
        self.calls = []
        self.response = FakeResponse()

    def __call__(self, url, *, json, headers, timeout):  # noqa: ANN001
        self.calls.append(
            {
                "url": url,
                "json": json,
                "headers": headers,
                "timeout": timeout,
            }
        )
        return self.response


def test_from_env_requires_brevo_api_key(monkeypatch) -> None:
    monkeypatch.delenv("BREVO_API_KEY", raising=False)

    try:
        BrevoHTTPSAdapter.from_env()
    except ValueError as exc:
        assert "BREVO_API_KEY" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_send_posts_brevo_transactional_email_with_csv_attachment(monkeypatch) -> None:
    fake_post = FakeHTTPPost()
    monkeypatch.setenv("BREVO_API_KEY", "brevo-secret")
    monkeypatch.delenv("EMAIL_FROM_NAME", raising=False)
    adapter = BrevoHTTPSAdapter.from_env(http_post=fake_post)
    message = MailMessage(
        sender="sender@example.com",
        recipient="owner@example.com",
        subject="Daily report",
        html_body="<p>Report</p>",
        attachment_filename="report.csv",
        attachment_content=b"column\r\nvalue\r\n",
    )

    adapter.send(message)

    request = fake_post.calls[0]
    assert request["url"] == "https://api.brevo.com/v3/smtp/email"
    assert request["headers"] == {
        "accept": "application/json",
        "api-key": "brevo-secret",
        "content-type": "application/json",
    }
    assert request["timeout"] == 30
    assert request["json"] == {
        "sender": {"email": "sender@example.com"},
        "to": [{"email": "owner@example.com"}],
        "subject": "Daily report",
        "htmlContent": "<p>Report</p>",
        "attachment": [
            {
                "content": base64.b64encode(b"column\r\nvalue\r\n").decode("ascii"),
                "name": "report.csv",
            }
        ],
    }
    assert fake_post.response.raise_for_status_called is True


def test_from_env_uses_optional_sender_display_name(monkeypatch) -> None:
    fake_post = FakeHTTPPost()
    monkeypatch.setenv("BREVO_API_KEY", "brevo-secret")
    monkeypatch.setenv("EMAIL_FROM_NAME", "Calls analysis")
    adapter = BrevoHTTPSAdapter.from_env(http_post=fake_post)
    message = MailMessage(
        sender="sender@example.com",
        recipient="owner@example.com",
        subject="Daily report",
        html_body="<p>Report</p>",
        attachment_filename="report.csv",
        attachment_content=b"column\r\nvalue\r\n",
    )

    adapter.send(message)

    assert fake_post.calls[0]["json"]["sender"] == {
        "email": "sender@example.com",
        "name": "Calls analysis",
    }


def test_send_delivers_to_multiple_comma_separated_recipients(monkeypatch) -> None:
    fake_post = FakeHTTPPost()
    monkeypatch.setenv("BREVO_API_KEY", "brevo-secret")
    monkeypatch.delenv("EMAIL_FROM_NAME", raising=False)
    adapter = BrevoHTTPSAdapter.from_env(http_post=fake_post)
    message = MailMessage(
        sender="sender@example.com",
        recipient="alice@example.com, bob@example.com; carol@example.com",
        subject="Daily report",
        html_body="<p>Report</p>",
        attachment_filename="report.csv",
        attachment_content=b"col\r\nval\r\n",
    )

    adapter.send(message)

    to_field = fake_post.calls[0]["json"]["to"]
    assert to_field == [
        {"email": "alice@example.com"},
        {"email": "bob@example.com"},
        {"email": "carol@example.com"},
    ]

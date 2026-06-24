from __future__ import annotations

from calls_analyser.adapters.mail.gmail import GmailSMTPAdapter
from calls_analyser.ports.mail import MailMessage


class FakeSMTP:
    instances = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logged_in = None
        self.sent_message = None
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def login(self, username: str, password: str) -> None:
        self.logged_in = (username, password)

    def send_message(self, message) -> None:  # noqa: ANN001
        self.sent_message = message


def test_from_env_requires_app_password(monkeypatch) -> None:
    monkeypatch.delenv("GOOGLE_app", raising=False)

    try:
        GmailSMTPAdapter.from_env()
    except ValueError as exc:
        assert "GOOGLE_app" in str(exc)
    else:
        raise AssertionError("Expected ValueError")


def test_send_uses_gmail_ssl_and_attaches_csv(monkeypatch) -> None:
    FakeSMTP.instances.clear()
    monkeypatch.setenv("GOOGLE_app", "abcd efgh ijkl mnop")
    adapter = GmailSMTPAdapter.from_env(smtp_factory=FakeSMTP)
    message = MailMessage(
        sender="tuttstt@gmail.com",
        recipient="owner@example.com",
        subject="Daily report",
        html_body="<p>Report</p>",
        attachment_filename="report.csv",
        attachment_content=b"column\r\nvalue\r\n",
    )

    adapter.send(message)

    smtp = FakeSMTP.instances[0]
    assert (smtp.host, smtp.port, smtp.timeout) == ("smtp.gmail.com", 465, 30)
    assert smtp.logged_in == ("tuttstt@gmail.com", "abcdefghijklmnop")
    assert smtp.sent_message["From"] == "tuttstt@gmail.com"
    assert smtp.sent_message["To"] == "owner@example.com"
    assert smtp.sent_message["Subject"] == "Daily report"
    attachment = next(smtp.sent_message.iter_attachments())
    assert attachment.get_filename() == "report.csv"
    assert attachment.get_content_type() == "text/csv"
    assert attachment.get_payload(decode=True) == b"column\r\nvalue\r\n"


def test_send_uses_comma_joined_to_header_for_multiple_recipients(monkeypatch) -> None:
    FakeSMTP.instances.clear()
    monkeypatch.setenv("GOOGLE_app", "abcd efgh ijkl mnop")
    adapter = GmailSMTPAdapter.from_env(smtp_factory=FakeSMTP)
    message = MailMessage(
        sender="tuttstt@gmail.com",
        recipient="alice@example.com, bob@example.com; carol@example.com",
        subject="Daily report",
        html_body="<p>Report</p>",
        attachment_filename="report.csv",
        attachment_content=b"column\r\nvalue\r\n",
    )

    adapter.send(message)

    smtp = FakeSMTP.instances[0]
    assert smtp.sent_message["To"] == "alice@example.com, bob@example.com, carol@example.com"

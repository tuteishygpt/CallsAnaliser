"""Gmail SMTP adapter."""
from __future__ import annotations

from email.message import EmailMessage
import os
import smtplib
from typing import Callable

from calls_analyser.ports.mail import MailMessage


GMAIL_ADDRESS = "tuttstt@gmail.com"


class GmailSMTPAdapter:
    """Deliver HTML reports through Gmail's SSL SMTP endpoint."""

    def __init__(
        self,
        *,
        username: str,
        app_password: str,
        smtp_factory: Callable[..., object] = smtplib.SMTP_SSL,
    ) -> None:
        self._username = username
        self._app_password = app_password
        self._smtp_factory = smtp_factory

    @classmethod
    def from_env(
        cls,
        *,
        smtp_factory: Callable[..., object] = smtplib.SMTP_SSL,
    ) -> "GmailSMTPAdapter":
        app_password = "".join(os.environ.get("GOOGLE_app", "").split())
        if not app_password:
            raise ValueError("GOOGLE_app is not configured.")
        return cls(
            username=GMAIL_ADDRESS,
            app_password=app_password,
            smtp_factory=smtp_factory,
        )

    def send(self, message: MailMessage) -> None:
        email = EmailMessage()
        email["From"] = message.sender
        email["To"] = message.recipient
        email["Subject"] = message.subject
        email.set_content("This report requires an HTML-capable email client.")
        email.add_alternative(message.html_body, subtype="html")
        email.add_attachment(
            message.attachment_content,
            maintype="text",
            subtype="csv",
            filename=message.attachment_filename,
        )

        with self._smtp_factory("smtp.gmail.com", 465, timeout=30) as smtp:
            smtp.login(self._username, self._app_password)
            smtp.send_message(email)

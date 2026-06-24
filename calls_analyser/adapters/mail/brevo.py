"""Brevo HTTPS mail adapter."""
from __future__ import annotations

import base64
import os
import re
from typing import Callable

import requests

from calls_analyser.ports.mail import MailMessage


BREVO_SEND_EMAIL_URL = "https://api.brevo.com/v3/smtp/email"


class BrevoHTTPSAdapter:
    """Deliver HTML reports through Brevo's transactional email HTTPS API."""

    def __init__(
        self,
        *,
        api_key: str,
        sender_name: str = "",
        http_post: Callable[..., object] = requests.post,
    ) -> None:
        self._api_key = api_key
        self._sender_name = sender_name
        self._http_post = http_post

    @classmethod
    def from_env(
        cls,
        *,
        http_post: Callable[..., object] = requests.post,
    ) -> "BrevoHTTPSAdapter":
        api_key = os.environ.get("BREVO_API_KEY", "").strip()
        if not api_key:
            raise ValueError("BREVO_API_KEY is not configured.")
        return cls(
            api_key=api_key,
            sender_name=os.environ.get("EMAIL_FROM_NAME", "").strip(),
            http_post=http_post,
        )

    def send(self, message: MailMessage) -> None:
        attachment_content = base64.b64encode(message.attachment_content).decode("ascii")
        sender = {"email": message.sender}
        if self._sender_name:
            sender["name"] = self._sender_name

        recipients = [r.strip() for r in re.split(r"[,;]+", message.recipient) if r.strip()]
        payload = {
            "sender": sender,
            "to": [{"email": r} for r in recipients],
            "subject": message.subject,
            "htmlContent": message.html_body,
            "attachment": [
                {
                    "content": attachment_content,
                    "name": message.attachment_filename,
                }
            ],
        }
        response = self._http_post(
            BREVO_SEND_EMAIL_URL,
            json=payload,
            headers={
                "accept": "application/json",
                "api-key": self._api_key,
                "content-type": "application/json",
            },
            timeout=30,
        )
        response.raise_for_status()

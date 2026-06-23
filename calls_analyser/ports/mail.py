"""Outbound email abstractions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class MailMessage:
    """Transport-neutral email with one CSV attachment."""

    sender: str
    recipient: str
    subject: str
    html_body: str
    attachment_filename: str
    attachment_content: bytes


class MailPort(Protocol):
    """Port implemented by outbound email providers."""

    def send(self, message: MailMessage) -> None:
        """Deliver an email message."""

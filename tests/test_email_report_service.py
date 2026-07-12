from __future__ import annotations

import pandas as pd

from calls_analyser.services.batch_results import EXPORT_RESULT_COLUMNS
from calls_analyser.services.email_report import EmailReportService


class RecordingMailPort:
    def __init__(self) -> None:
        self.messages = []

    def send(self, message) -> None:  # noqa: ANN001
        self.messages.append(message)


def _results() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Start": "2026-06-22T09:00:00",
                "Caller": "<Client A>",
                "Destination": "Support",
                "Duration (s)": 120,
                "UniqueId": "call-1",
                "user": "operator-1",
                "Needs follow-up": "Yes",
                "Reason": "Call back & clarify",
                "Link": '<a href="https://example.test/call-1?a=1&amp;b=2" target="_blank">Listen</a>',
                "Status": "✅",
            },
            {
                "Start": "2026-06-22T10:00:00",
                "Caller": "Client B",
                "Destination": "Sales",
                "Duration (s)": 60,
                "UniqueId": "call-2",
                "Needs follow-up": "No",
                "Reason": "Resolved",
                "Link": "",
                "Status": "✅",
            },
        ]
    )


def test_send_filters_html_but_attaches_complete_csv() -> None:
    mail = RecordingMailPort()
    service = EmailReportService(mail, sender="tuttstt@gmail.com", recipient="owner@example.com")

    service.send(
        _results(),
        filter_option="Needs follow-up",
        report_date="2026-06-22",
        tenant_id="lix",
    )

    message = mail.messages[0]
    assert message.recipient == "owner@example.com"
    assert "Calls analysis — lix — 2026-06-22" == message.subject
    assert "call-1" in message.html_body
    assert "call-2" not in message.html_body
    assert "UniqueId" not in message.html_body
    assert "2026-06-22 09:00:00" in message.html_body
    assert "operator-1" in message.html_body
    assert "&lt;Client A&gt;" in message.html_body
    assert "Call back &amp; clarify" in message.html_body
    assert 'href="https://example.test/call-1?a=1&amp;b=2"' in message.html_body

    csv_text = message.attachment_content.decode("utf-8-sig")
    assert "UniqueId" not in csv_text
    assert "call-1" in csv_text
    assert "call-2" not in csv_text
    assert "2026-06-22 09:00:00" in csv_text
    assert "operator-1" in csv_text
    assert message.attachment_filename == "calls-analysis-lix-2026-06-22.csv"


def test_send_rejects_empty_results() -> None:
    mail = RecordingMailPort()
    service = EmailReportService(mail, sender="tuttstt@gmail.com", recipient="owner@example.com")

    try:
        service.send(
            pd.DataFrame(),
            filter_option="All",
            report_date="2026-06-22",
            tenant_id="lix",
        )
    except ValueError as exc:
        assert str(exc) == "No batch results to send."
    else:
        raise AssertionError("Expected ValueError")


def test_send_allows_empty_filtered_html_and_keeps_complete_csv() -> None:
    mail = RecordingMailPort()
    service = EmailReportService(mail, sender="tuttstt@gmail.com", recipient="owner@example.com")

    service.send(
        _results(),
        filter_option="No follow-up",
        report_date="2026-06-22",
        tenant_id="lix",
    )

    message = mail.messages[0]
    assert "Client B" in message.html_body
    assert "Resolved" in message.html_body
    assert "call-1" not in message.html_body
    assert "call-1" in message.attachment_content.decode("utf-8-sig")


def test_send_does_not_make_non_http_link_clickable() -> None:
    mail = RecordingMailPort()
    service = EmailReportService(mail, sender="tuttstt@gmail.com", recipient="owner@example.com")
    results = _results()
    results.loc[0, "Link"] = '<a href="javascript:alert(1)">Listen</a>'

    service.send(
        results,
        filter_option="Needs follow-up",
        report_date="2026-06-22",
        tenant_id="lix",
    )

    html_body = mail.messages[0].html_body
    assert "javascript:" not in html_body
    assert ">Listen<" not in html_body


def test_report_orders_final_columns_before_audit_columns() -> None:
    results = _results().assign(
        **{
            "Initial needs follow-up": "Yes",
            "Initial reason": "Initial",
            "Verification needs follow-up": "No",
            "Verification reason": "Verified",
            "Verification status": "complete",
        }
    )

    report = EmailReportService._prepare_report_results(results)

    assert list(report.columns) == EXPORT_RESULT_COLUMNS

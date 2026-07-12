"""Build and send batch-analysis email reports."""
from __future__ import annotations

from html import escape, unescape
import re
from urllib.parse import urlsplit

import pandas as pd

from calls_analyser.ports.mail import MailMessage, MailPort
from calls_analyser.ui.utils import prepare_results_display


REPORT_COLUMNS = [
    "Start",
    "Caller",
    "Destination",
    "user",
    "Duration (s)",
    "Needs follow-up",
    "Reason",
    "Initial needs follow-up",
    "Initial reason",
    "Verification needs follow-up",
    "Verification reason",
    "Verification status",
    "Link",
    "Status",
]
_LINK_PATTERN = re.compile(r"""href=["']([^"']+)["']""", re.IGNORECASE)


class EmailReportService:
    """Format batch results and delegate delivery to a mail adapter."""

    def __init__(self, mail_port: MailPort, *, sender: str, recipient: str) -> None:
        self._mail_port = mail_port
        self._sender = sender
        self._recipient = recipient

    def send(
        self,
        results: pd.DataFrame,
        *,
        filter_option: str,
        report_date: str,
        tenant_id: str,
    ) -> None:
        if results is None or results.empty:
            raise ValueError("No batch results to send.")

        full_results = self._prepare_report_results(results)
        visible_results = self._filter_results(full_results, filter_option)
        safe_tenant = re.sub(r"[^A-Za-z0-9_.-]+", "-", tenant_id).strip("-") or "tenant"
        safe_date = re.sub(r"[^0-9-]+", "-", report_date).strip("-") or "report"

        message = MailMessage(
            sender=self._sender,
            recipient=self._recipient,
            subject=f"Calls analysis — {tenant_id} — {report_date}",
            html_body=self._render_html(
                visible_results,
                filter_option=filter_option,
                report_date=report_date,
                tenant_id=tenant_id,
                total_count=len(full_results),
            ),
            attachment_filename=f"calls-analysis-{safe_tenant}-{safe_date}.csv",
            attachment_content=full_results.to_csv(index=False).encode("utf-8-sig"),
        )
        self._mail_port.send(message)

    @staticmethod
    def _prepare_report_results(results: pd.DataFrame) -> pd.DataFrame:
        columns = [column for column in REPORT_COLUMNS if column in results.columns]
        report = prepare_results_display(results)
        extras = [column for column in report.columns if column not in columns]
        return report.reindex(columns=columns + extras, fill_value="")

    @staticmethod
    def _filter_results(results: pd.DataFrame, filter_option: str) -> pd.DataFrame:
        if filter_option == "Needs follow-up":
            return results[results["Needs follow-up"] == "Yes"]
        if filter_option == "No follow-up":
            return results[results["Needs follow-up"] == "No"]
        return results

    @classmethod
    def _render_html(
        cls,
        results: pd.DataFrame,
        *,
        filter_option: str,
        report_date: str,
        tenant_id: str,
        total_count: int,
    ) -> str:
        headers = "".join(
            f'<th style="{cls._header_style()}">{escape(column)}</th>'
            for column in results.columns
        )
        rows = "".join(
            "<tr>"
            + "".join(
                f'<td style="{cls._cell_style()}">{cls._render_cell(column, row[column])}</td>'
                for column in results.columns
            )
            + "</tr>"
            for _, row in results.iterrows()
        )
        if not rows:
            rows = (
                f'<tr><td colspan="{len(results.columns)}" style="{cls._cell_style()}">'
                "No rows match the selected filter.</td></tr>"
            )

        return (
            "<html><body>"
            f"<h2>Calls analysis: {escape(tenant_id)} — {escape(report_date)}</h2>"
            f"<p>Filter: <strong>{escape(filter_option)}</strong>. "
            f"Shown: {len(results)} of {total_count}. "
            "The attached CSV contains all results.</p>"
            '<div style="overflow-x:auto">'
            '<table style="border-collapse:collapse;font-family:Arial,sans-serif;font-size:13px">'
            f"<thead><tr>{headers}</tr></thead><tbody>{rows}</tbody></table></div>"
            "</body></html>"
        )

    @staticmethod
    def _render_cell(column: str, value: object) -> str:
        text = "" if pd.isna(value) else str(value)
        if column == "Link":
            match = _LINK_PATTERN.search(text)
            if match:
                raw_url = unescape(match.group(1))
                if urlsplit(raw_url).scheme not in {"http", "https"}:
                    return ""
                url = escape(raw_url, quote=True)
                return f'<a href="{url}" target="_blank" rel="noopener noreferrer">Listen</a>'
        return escape(text)

    @staticmethod
    def _header_style() -> str:
        return "background:#4f46e5;color:#fff;border:1px solid #ddd;padding:8px;text-align:left"

    @staticmethod
    def _cell_style() -> str:
        return "border:1px solid #ddd;padding:8px;vertical-align:top"

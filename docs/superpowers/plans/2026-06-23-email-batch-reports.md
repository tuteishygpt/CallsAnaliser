# Email Batch Reports Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add manual and scheduled Gmail delivery of batch results as a filtered HTML table with a complete CSV attachment.

**Architecture:** A report service owns filtering and document generation, while a Gmail adapter owns SMTP transport behind a mail port. UI and scheduler entry points share these components.

**Tech Stack:** Python 3.10+, standard-library `email`/`smtplib`, pandas, Gradio, pytest.

---

## Chunk 1: Report generation and Gmail transport

### Task 1: Define mail transport and report service

**Files:**
- Create: `calls_analyser/ports/mail.py`
- Create: `calls_analyser/services/email_report.py`
- Test: `tests/test_email_report_service.py`

- [ ] Write failing tests for HTML filtering, escaping, safe links, and complete CSV attachment data.
- [ ] Run `pytest tests/test_email_report_service.py -v` and confirm failures are caused by missing production code.
- [ ] Implement `MailMessage`, `MailPort`, and `EmailReportService`.
- [ ] Re-run the focused tests and confirm they pass.

### Task 2: Implement Gmail SMTP adapter

**Files:**
- Create: `calls_analyser/adapters/mail/__init__.py`
- Create: `calls_analyser/adapters/mail/gmail.py`
- Test: `tests/test_gmail_mail_adapter.py`

- [ ] Write failing tests for configuration validation, SMTP SSL login, HTML body, and CSV attachment.
- [ ] Run `pytest tests/test_gmail_mail_adapter.py -v` and confirm expected failures.
- [ ] Implement `GmailSMTPAdapter` using `GOOGLE_app`, `EMAIL_TO`, and `SMTP_SSL`.
- [ ] Re-run focused tests and confirm they pass.

## Chunk 2: UI and scheduler integration

### Task 3: Add manual UI delivery

**Files:**
- Modify: `calls_analyser/ui/dependencies.py`
- Modify: `calls_analyser/ui/handlers.py`
- Modify: `calls_analyser/ui/layout.py`
- Test: `tests/test_app_batch.py`

- [ ] Write a failing test proving the selected UI filter controls HTML rows while CSV remains complete.
- [ ] Run the focused test and confirm the missing handler behavior causes failure.
- [ ] Wire the report service, add the button, and return a user-facing status.
- [ ] Re-run UI batch tests and confirm they pass.

### Task 4: Add scheduled delivery

**Files:**
- Modify: `calls_analyser/runner.py`
- Modify: `app.py`
- Test: `tests/test_runner_email.py`

- [ ] Write failing tests proving successful scheduled batches send all rows and default HTML to follow-up rows.
- [ ] Run focused tests and confirm failures.
- [ ] Make `run_batch_process` return complete rows and have the scheduler send them after analysis.
- [ ] Re-run focused tests and confirm they pass.

## Chunk 3: Documentation and verification

### Task 5: Document configuration and verify

**Files:**
- Modify: `README.md`

- [ ] Document `GOOGLE_app`, `EMAIL_TO`, sender behavior, and automatic/manual delivery.
- [ ] Run `pytest`.
- [ ] Inspect `git diff --check`.
- [ ] Review the final diff for secret leakage and unrelated changes.

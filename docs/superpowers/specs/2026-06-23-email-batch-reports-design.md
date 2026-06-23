# Email Batch Reports Design

## Goal

Send batch-analysis results through Gmail SMTP both manually from the Gradio UI
and automatically after the scheduled daily batch.

## User-visible behavior

- The UI adds a `Send by email` button beside `Save to file`.
- The existing result filter (`All`, `Needs follow-up`, `No follow-up`) controls
  which rows appear in the HTML table in the email.
- The attached CSV always contains the complete, unfiltered batch result.
- Scheduled reports show `Needs follow-up` rows in the HTML body and attach all
  rows as CSV.
- The sender is `tuttstt@gmail.com`.
- The recipient comes from `EMAIL_TO`, falling back to the sender address.
- The Gmail app password comes from `GOOGLE_app`.

## Architecture

`EmailReportService` converts result rows into a safe HTML table and UTF-8 CSV,
then delegates delivery to a mail port. `GmailSMTPAdapter` implements the port
with `SMTP_SSL` on `smtp.gmail.com:465`. UI handlers and the daily runner call
the same report service, keeping message formatting and security rules in one
place.

## Data and security

The report uses the final UI columns: `Start`, `Caller`, `Destination`,
`Duration (s)`, `UniqueId`, `Needs follow-up`, `Reason`, `Link`, and `Status`.
Cell values are HTML-escaped. The existing `Listen` anchor is converted into a
safe link rather than copied as arbitrary HTML. SMTP credentials are read only
from environment variables and are never logged.

## Error handling

Missing mail configuration produces a clear UI/scheduler error. SMTP failures
do not remove analysis results or cache entries. The scheduled batch logs email
failure after analysis has completed.

## Testing

Unit tests cover filtering, HTML escaping, CSV completeness, Gmail message
construction, missing configuration, UI delegation, and scheduler delivery.
SMTP network calls are replaced with injected fakes.

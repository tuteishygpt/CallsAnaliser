# Calls Analyser

A hexagonal architecture Python project for analysing and summarising call transcripts.

## Gemini BATCH mode

To enable Gemini BATCH processing for mass analysis, adjust `batch_params.json` in the project root:

```json
{
  "enable_gemini_batch": true,
  "batch_size": 25
}
```

- `enable_gemini_batch`: set to `true` to send batch jobs through the Gemini BATCH API instead of per-call requests.
- `batch_size`: how many recordings are packed into a single Gemini BATCH job (minimum 1).

# Saved Analysis Comparison Implementation Plan

> **For agentic workers:** Implement directly in the current session. The user explicitly waived automated tests.

**Goal:** Export a read-only Excel comparison of all saved analysis variants for filtered calls.

**Architecture:** A Python CLI obtains filtered call IDs and paginates read-only Supabase selects, then pivots the records into JSON. A bundled-runtime JavaScript builder converts that JSON into a formatted, verified workbook.

**Tech Stack:** Python, existing Calls Analyser services, Supabase/PostgREST, `@oai/artifact-tool`.

---

## Chunk 1: Script and workbook

- [ ] Add the read-only Python CLI with deterministic variant labels and pagination.
- [ ] Add the artifact-tool workbook builder.
- [ ] Run the CLI for Amedis, 2026-07-10, 19:00–20:00, Inbound.
- [ ] Inspect values and formula errors, render both sheets, and visually verify them.

## Chunk 2: Parsed decisions

- [ ] Parse plain and fenced JSON from every model-result column.
- [ ] Add per-model `needs_follow_up` and `reason` columns while retaining raw JSON.
- [ ] Add and color-code `MATCH`, `DIFFERENT`, and `MISSING/INVALID` decision comparison.
- [ ] Regenerate and inspect the workbook.

## Chunk 3: Audio links and compact comparison

- [ ] Keep raw model JSON only on `Raw Data`.
- [ ] Add a clickable `Audio` link to each `Comparison` row without downloading audio.
- [ ] Regenerate the all-day workbook and verify exported links and comparison values.

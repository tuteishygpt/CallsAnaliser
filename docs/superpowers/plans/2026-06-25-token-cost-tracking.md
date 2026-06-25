# Token Cost Tracking Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Store Gemini token usage and estimated internal/client token costs for each newly processed call.

**Architecture:** Add a small usage-tracking service that receives call context, model context, execution mode, and `usageMetadata`; it resolves pricing from Supabase `model_pricing`, calculates snapshots, and inserts one row into `analysis_usage`. Direct UI paths keep using `AnalysisService`; Vertex batch paths return text plus usage metadata and record usage after successful uncached results.

**Tech Stack:** Python, Pydantic/dataclasses, google-genai response metadata, Supabase PostgREST client, pytest.

---

## Chunk 1: Usage Model And Pricing

### Task 1: Usage pricing service

**Files:**
- Create: `calls_analyser/services/usage.py`
- Test: `tests/test_usage_tracking.py`

- [ ] Write failing tests for token extraction and cost/client price calculation.
- [ ] Implement minimal dataclasses/functions for usage metadata, pricing snapshots, and estimated amounts.
- [ ] Run `pytest tests/test_usage_tracking.py -v`.

## Chunk 2: Supabase Persistence

### Task 2: Supabase usage adapter

**Files:**
- Create: `calls_analyser/adapters/storage/supabase_usage.py`
- Modify: `calls_analyser/ui/dependencies.py`
- Test: `tests/test_supabase_usage.py`

- [ ] Write failing tests that pricing is read from `model_pricing` and rows are inserted into `analysis_usage`.
- [ ] Implement Supabase adapter with graceful no-pricing/no-client handling.
- [ ] Wire adapter into app dependencies when Supabase credentials exist.

## Chunk 3: Gemini Metadata Capture

### Task 3: Direct and batch metadata

**Files:**
- Modify: `calls_analyser/adapters/ai/gemini.py`
- Modify: `calls_analyser/services/gemini_batch.py`
- Test: `tests/test_gemini_usage_metadata.py`
- Test: existing batch tests

- [ ] Write failing tests for `usageMetadata` extraction from direct and batch responses.
- [ ] Store direct response usage under `AnalysisResult.metadata["usage_metadata"]`.
- [ ] Add `VertexBatchRunner.run_batch_results()` returning text plus usage metadata while preserving `run_batch()` string behavior.

## Chunk 4: Usage Recording

### Task 4: Wire write paths

**Files:**
- Modify: `calls_analyser/services/analysis.py`
- Modify: `calls_analyser/ui/handlers.py`
- Modify: `calls_analyser/runner.py`
- Test: focused existing UI/runner tests plus new usage tests

- [ ] Add optional `usage_tracker` dependency to `AnalysisService`.
- [ ] Record `mode="ui_direct"` for single-call UI and `mode="ui_mass"` for mass UI calls via `AnalysisOptions`.
- [ ] Record `mode="scheduler_batch"` for Vertex batch scheduler results.
- [ ] Do not record cache hits as paid usage.

## Verification

- [ ] Run focused usage/Gemini/Supabase tests.
- [ ] Run the full pytest suite.

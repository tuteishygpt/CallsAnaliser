# UI Tenant Batch Model Design

## Goal

Make local Gradio mass analysis use the selected tenant's batch model and
batch language settings, while changing the global fallback model to
`models/gemini-3.1-flash-lite`.

## Scope

- Change the global `BATCH_MODEL_KEY` fallback to
  `models/gemini-3.1-flash-lite`.
- Resolve runtime settings for the selected tenant before mass analysis.
- Use a non-empty tenant `batch_model_key` in preference to the global model.
- Use a valid tenant `batch_language_code` in preference to the global batch
  language, matching the scheduler's fallback behavior.
- Keep direct, single-call analysis controlled by the UI model dropdown.
- Do not change scheduler behavior.

## Design

Add focused helpers to the UI handler for resolving tenant batch model and
language values. The mass-analysis path resolves the tenant once, resolves its
runtime settings once through `tenant_settings_service`, and passes the
resulting model and language through the existing sequential analysis loop.
Missing or blank tenant model values fall back to the application-wide model.
Missing, blank, or invalid tenant language values fall back to the global batch
language.

The resolved model key must be used consistently for registry validation and
`AnalysisOptions`; `AnalysisService` then uses that exact key for registry
lookup and cache identity. `AnalysisResult.model` remains provider-owned and
may be an expanded Vertex resource path rather than the registry key; this
change does not normalize that existing representation. Any legacy UI batch
helper reachable from mass analysis must receive the same resolved model key
instead of reading the global dependency again.

## Error Handling

If tenant settings cannot be resolved, mass analysis retains the current
global behavior instead of failing solely because optional tenant overrides
are unavailable. An invalid tenant language also falls back to the global
batch language. A non-empty tenant model that is not registered is treated as
a tenant configuration error: validate it once before fetching or iterating
calls and return one clear UI error rather than failing every call or silently
using a different model.

## Testing

- Verify mass analysis resolves tenant settings once and passes tenant
  `batch_model_key` and tenant language to the analysis service.
- Verify blank settings and a settings-service exception use the global model
  and language.
- Verify invalid tenant language uses the global language.
- Verify an unregistered non-empty tenant model fails once before call
  processing.
- At the real `AnalysisService`/registry/cache boundary, verify the tenant
  model selects the matching provider, appears in the cache key, and does not
  reuse a cached result created under a different model. Accept the provider's
  existing `AnalysisResult.model` representation without normalizing it.
- Verify direct single-call analysis still uses the model selected in the UI
  dropdown even when tenant batch settings contain another model.
- Verify the configured global batch model is
  `models/gemini-3.1-flash-lite`.
- Do not edit scheduler or runner model-resolution logic. The shared global
  constant change intentionally changes only their fallback when a tenant has
  no batch model override; existing scheduler tests must remain unchanged and
  pass.
- Run the complete test suite after the focused tests.

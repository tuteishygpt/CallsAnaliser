# Saved Analysis Comparison Script

## Goal

Create and run a one-off script that compares already-saved call analyses for all models and prompt variants. The script must not invoke an AI model, download recordings, mutate Supabase, or change the Gradio UI.

## Selection

The initial run uses these call filters:

- tenant: Amedis (`amedis` tenant identifier resolved by existing configuration);
- date: 2026-07-10;
- time: 19:00 inclusive through 20:00 inclusive, using the existing call-list filtering semantics;
- call type: Inbound.

The script obtains matching calls through the existing tenant and call-log services. It then queries `analysis_results` for the resolved tenant and matching `call_unique_id` values. Results are not filtered by model, provider, prompt key, prompt version, or custom prompt.

## Output

The script writes one `.xlsx` workbook with two worksheets:

1. `Comparison`: one row per `UniqueId`, call metadata columns, and one dynamic column per distinct analysis variant. A variant is identified by provider, model, prompt key, prompt version, and custom prompt. Column labels use readable model/prompt/version names and a shortened custom-prompt fragment; deterministic suffixes resolve label collisions.
2. `Raw Data`: one row per saved analysis record, including the full cache identity, result text, metadata, and available creation timestamp. This preserves every source record for auditing and filtering.

Calls without saved analysis records are excluded from `Comparison`. If there are no matching calls or no saved analyses, the script still creates a valid workbook with headers and an explanatory status row.

## Components and Data Flow

- A read-only Supabase repository method fetches saved analysis records in bounded chunks by tenant and call IDs.
- A comparison service converts call entries plus saved records into stable rectangular data for both worksheets.
- A command-line script accepts filter/output arguments, uses existing application dependency wiring, and delegates workbook creation to a small JavaScript builder using the bundled `@oai/artifact-tool` runtime.
- The builder formats headers, enables filters/frozen headers, wraps long analysis text, and exports the workbook.

## Error Handling

Missing Supabase configuration, an unknown tenant, telephony lookup failure, and workbook export failure terminate with a clear non-zero CLI error. The script reports counts of matching calls, saved records, represented IDs, and output path. It never falls back to running analysis.

## Testing and Verification

- Unit tests cover repository query scoping/chunking and pivot behavior, including multiple variants for one ID and label collisions.
- A CLI-level test verifies filters are passed correctly and no analysis service is called.
- The real run verifies workbook values, scans for formula errors, and renders both worksheets for visual inspection before delivery.

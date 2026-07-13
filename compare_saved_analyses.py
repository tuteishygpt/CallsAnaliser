"""Export already-saved call analyses to a side-by-side Excel workbook."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from calls_analyser.env import load_environment


def _variant_label(record: dict) -> str:
    model = str(record.get("model_key") or record.get("provider_name") or "unknown")
    prompt = str(record.get("prompt_key") or "unknown")
    version = int(record.get("prompt_version") or 1)
    custom = str(record.get("custom_fragment") or "").strip()
    label = f"{model} · {prompt} · v{version}"
    if custom:
        compact = " ".join(custom.split())
        digest = hashlib.sha1(custom.encode("utf-8")).hexdigest()[:8]
        label += f" · custom: {compact[:48]} [{digest}]"
    return label


def _parse_analysis_json(text: str) -> tuple[bool | None, str]:
    """Return needs_follow_up and reason from plain or fenced model JSON."""
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()
    left, right = cleaned.find("{"), cleaned.rfind("}")
    if left >= 0 and right > left:
        cleaned = cleaned[left : right + 1]
    try:
        payload = json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None, ""
    decision = payload.get("needs_follow_up")
    reason = str(payload.get("reason") or "")
    return (decision if isinstance(decision, bool) else None), reason


def _fetch_saved(table, tenant_id: str, unique_ids: list[str]) -> list[dict]:
    records: list[dict] = []
    page_size = 1000
    for start in range(0, len(unique_ids), 100):
        chunk = unique_ids[start : start + 100]
        offset = 0
        while True:
            response = (
                table.select("*")
                .eq("tenant_id", tenant_id)
                .in_("call_unique_id", chunk)
                .order("call_unique_id")
                .range(offset, offset + page_size - 1)
                .execute()
            )
            page = list(response.data or [])
            records.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
    return records


def _audio_url(entry, tenant) -> str:  # noqa: ANN001
    raw = entry.raw or {}
    for key in ("record", "recording_url", "download_url"):
        value = str(raw.get(key) or "").strip()
        if value:
            return value
    return tenant.recording_url(str(entry.unique_id))


def _rows(entries, records: list[dict], tenant) -> tuple[list[list], list[list]]:  # noqa: ANN001
    by_uid = {}
    for entry in entries:
        by_uid.setdefault(str(entry.unique_id), entry)
    labels = sorted({_variant_label(record) for record in records})
    record_map = {(str(r.get("call_unique_id")), _variant_label(r)): str(r.get("result_text") or "") for r in records}
    represented = sorted({str(r.get("call_unique_id")) for r in records if str(r.get("call_unique_id")) in by_uid})
    parsed_headers = [
        header
        for label in labels
        for header in (f"{label} | needs_follow_up", f"{label} | reason")
    ]
    comparison = [[
        "UniqueId", "Audio", "Start", "Caller", "Destination", "Duration (s)",
        *parsed_headers, "needs_follow_up comparison",
    ]]
    for uid in represented:
        entry = by_uid[uid]
        raw_values = [record_map.get((uid, label), "") for label in labels]
        parsed = [_parse_analysis_json(value) for value in raw_values]
        decisions = [decision for decision, _reason in parsed]
        if len(decisions) < 2 or any(decision is None for decision in decisions):
            decision_comparison = "MISSING/INVALID"
        elif len(set(decisions)) == 1:
            decision_comparison = "MATCH"
        else:
            decision_comparison = "DIFFERENT"
        comparison.append([
            uid,
            _audio_url(entry, tenant),
            entry.started_at.isoformat() if entry.started_at else "",
            entry.caller_id or "",
            entry.destination or "",
            entry.duration_seconds,
            *[
                value
                for decision, reason in parsed
                for value in ("TRUE" if decision is True else "FALSE" if decision is False else "", reason)
            ],
            decision_comparison,
        ])
    raw_headers = ["call_unique_id", "tenant_id", "provider_name", "model_key", "prompt_key", "prompt_version", "custom_fragment", "created_at", "result_text", "metadata", "variant_label"]
    raw = [raw_headers]
    for record in sorted(records, key=lambda r: (str(r.get("call_unique_id")), _variant_label(r), str(r.get("created_at") or ""))):
        raw.append([json.dumps(record.get(key), ensure_ascii=False) if key == "metadata" else record.get(key, "") for key in raw_headers[:-1]] + [_variant_label(record)])
    return comparison, raw


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tenant", default="Amedis")
    parser.add_argument("--date", default="2026-07-10")
    parser.add_argument("--time-from", default="19:00")
    parser.add_argument("--time-to", default="20:00")
    parser.add_argument("--call-type", type=int, default=0, help="0=Inbound, 1=Outbound, 2=Internal")
    parser.add_argument("--output", default="outputs/saved-analysis-comparison/amedis_2026-07-10_1900-2000.xlsx")
    args = parser.parse_args()

    load_environment()
    from calls_analyser.ui.dependencies import build_dependencies
    from calls_analyser.ui.utils import parse_day, parse_time_value

    deps = build_dependencies()
    tenant = deps.tenant_service.resolve(args.tenant)
    entries = deps.call_log_service.list_calls(parse_day(args.date), tenant, time_from=parse_time_value(args.time_from), time_to=parse_time_value(args.time_to), call_type=args.call_type)
    entries = list({str(entry.unique_id): entry for entry in entries}.values())
    cache = getattr(deps.analysis_service, "_cache", None)
    table = getattr(cache, "_table", None)
    if table is None:
        raise RuntimeError("Supabase analysis_results is unavailable; check SUPABASE_URL and SUPABASE_KEY")
    records = _fetch_saved(table, tenant.tenant_id, [str(entry.unique_id) for entry in entries]) if entries else []
    comparison, raw = _rows(entries, records, tenant)

    output = Path(args.output).resolve()
    qa_dir = output.parent / "qa"
    with tempfile.TemporaryDirectory(prefix="calls-analysis-comparison-") as temp_dir:
        input_path = Path(temp_dir) / "input.json"
        input_path.write_text(json.dumps({"comparisonRows": comparison, "rawRows": raw}, ensure_ascii=False), encoding="utf-8")
        node = os.environ.get("CODEX_BUNDLED_NODE", r"C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")
        node_modules = os.environ.get("CODEX_BUNDLED_NODE_MODULES", r"C:\Users\admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules")
        builder = Path(temp_dir) / "build_saved_analysis_comparison.mjs"
        shutil.copy2(Path(__file__).parent / "scripts" / builder.name, builder)
        subprocess.run(["cmd", "/c", "mklink", "/J", str(Path(temp_dir) / "node_modules"), node_modules], check=True, capture_output=True)
        try:
            subprocess.run([node, str(builder), str(input_path), str(output), str(qa_dir)], check=True)
        except subprocess.CalledProcessError:
            previews = [qa_dir / "Comparison.png", qa_dir / "Raw_Data.png"]
            if not output.exists() or not all(path.exists() for path in previews):
                raise
            print("WARNING: workbook runtime exited abnormally after all verified artifacts were saved")
    print(f"Calls found: {len(entries)}")
    print(f"Saved analysis records: {len(records)}")
    print(f"UniqueIds represented: {max(0, len(comparison) - 1)}")
    print(f"Workbook: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

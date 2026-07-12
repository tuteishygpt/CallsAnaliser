"""Gradio event handlers separated from app wiring."""
from __future__ import annotations

import json
import os
import re
import tempfile
from types import SimpleNamespace
from urllib.parse import parse_qs, unquote, urlparse

import gradio as gr
import pandas as pd

from . import config
from .dependencies import AppDependencies, AnalysisOptions
from . import utils
from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
from calls_analyser.domain.models import AnalysisResult
from calls_analyser.services.batch_results import build_batch_item_row
from calls_analyser.services.gemini_batch import BatchTask, VertexBatchRunner, guess_mime_type
from calls_analyser.services.usage_report import (
    ALL_VALUE,
    UsageReportFilters,
    build_usage_report,
)


class UIHandlers:
    def __init__(self, deps: AppDependencies):
        self.deps = deps

    # ----------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------
    @staticmethod
    def _parse_follow_up_fields(text: str) -> tuple[str, str]:
        text_clean = str(text or "").strip()
        # Strip markdown code fences that models sometimes wrap JSON in
        if text_clean.startswith("```"):
            lines = text_clean.splitlines()
            if lines[-1].strip() == "```":
                lines = lines[1:-1]
            elif lines[0].strip().startswith("```"):
                lines = lines[1:]
            text_clean = "\n".join(lines).strip()
        try:
            left, right = text_clean.find("{"), text_clean.rfind("}")
            if left != -1 and right != -1 and right > left:
                text_clean = text_clean[left : right + 1]
            payload = json.loads(text_clean)
            needs_follow_up = payload.get("needs_follow_up")
            needs_str = "Yes" if needs_follow_up else "No"
            reason = str(payload.get("reason") or "")
            return needs_str, reason
        except Exception as exc:
            print(f"DEBUG _parse_follow_up_fields failed: {exc}; text={text_clean!r:.300}")
            if "Needs follow-up:" in text_clean:
                try:
                    parts = text_clean.split("Summary:", 1)
                    nf_part = parts[0].replace("Needs follow-up:", "").strip()
                    summary_part = parts[1].strip() if len(parts) > 1 else ""
                    return nf_part, summary_part
                except Exception:
                    return "", text_clean
        return "", text_clean

    @staticmethod
    def _build_row_base(entry):
        raw = getattr(entry, "raw", {}) or {}
        user = raw.get("user")
        return {
            "Start": entry.started_at.isoformat() if entry.started_at else "",
            "Caller": entry.caller_id or "",
            "Destination": entry.destination or "",
            "Duration (s)": entry.duration_seconds,
            "UniqueId": entry.unique_id,
            **({"user": user} if user not in (None, "") else {}),
        }

    @staticmethod
    def _uid_from_row(row) -> str:
        raw = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
        for key in ("UniqueId", "uid", "Uid", "UID", "unique_id", "UniqueID", "id", "Id", "ID"):
            value = raw.get(key)
            if value not in (None, ""):
                return str(value).strip()

        link_value = raw.get("Link") or raw.get("recording_url") or raw.get("record")
        if not link_value:
            return ""
        link_text = str(link_value).strip()
        href_match = re.search(r"""href=["']([^"']+)["']""", link_text)
        href = href_match.group(1) if href_match else link_text
        parsed = urlparse(href)
        query = parse_qs(parsed.query)
        for key in ("unique_id", "uid", "id"):
            values = query.get(key)
            if values and values[0]:
                return str(values[0]).strip()
        last_segment = unquote(parsed.path.rstrip("/").rsplit("/", 1)[-1])
        if last_segment.lower().endswith(".mp3"):
            last_segment = last_segment[:-4]
        return last_segment.strip()

    @staticmethod
    def _recording_url_from_row(row) -> str:
        raw = row.to_dict() if hasattr(row, "to_dict") else dict(row or {})
        link_value = raw.get("Link") or raw.get("recording_url") or raw.get("record")
        if not link_value:
            return ""
        link_text = str(link_value).strip()
        href_match = re.search(r"""href=["']([^"']+)["']""", link_text)
        return (href_match.group(1) if href_match else link_text).strip()

    @staticmethod
    def _entry_from_row(unique_id: str, row):
        if row is None:
            return None
        raw = row.to_dict() if hasattr(row, "to_dict") else dict(row)

        def first_value(*keys):
            for key in keys:
                value = raw.get(key)
                if value not in (None, ""):
                    return value
            return None

        def as_int(value):
            try:
                if pd.isna(value):
                    return None
            except TypeError:
                pass
            try:
                return int(value)
            except (TypeError, ValueError):
                return None

        started_raw = first_value("Start", "start_time", "start", "started_at")
        started_at = None
        if started_raw:
            try:
                started_at = pd.to_datetime(started_raw).to_pydatetime()
            except Exception:
                started_at = None

        return SimpleNamespace(
            unique_id=unique_id,
            started_at=started_at,
            caller_id=first_value("Caller", "client", "phone_number", "caller_id"),
            destination=first_value("Destination", "destination"),
            duration_seconds=as_int(first_value("Duration (s)", "duration_seconds", "duration")),
            raw=raw,
        )

    @staticmethod
    def _find_row_by_unique_id(df, unique_id: str):
        if df is None or getattr(df, "empty", True):
            return None
        if "UniqueId" not in df.columns:
            return None
        matches = df[df["UniqueId"].astype(str) == str(unique_id)]
        if matches.empty:
            return None
        return matches.iloc[0]

    @staticmethod
    def _single_visible_row_without_unique_id(df):
        if df is None or getattr(df, "empty", True):
            return None
        if "UniqueId" in df.columns or len(df) != 1:
            return None
        return df.iloc[0]

    @staticmethod
    def _visual_row_index_from_select_event(evt: gr.SelectData):
        index = getattr(evt, "index", None)
        if isinstance(index, (list, tuple)):
            if not index:
                return None
            index = index[0]
        try:
            return int(index)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _find_batch_original_row(
        displayed_df: pd.DataFrame,
        full_df_state: pd.DataFrame,
        visual_row_index: int,
    ):
        clicked_row = displayed_df.iloc[visual_row_index]
        uid = UIHandlers._uid_from_row(clicked_row)
        if uid:
            if "UniqueId" not in full_df_state.columns:
                return clicked_row
            matches = full_df_state[full_df_state["UniqueId"].astype(str) == uid]
            return matches.iloc[0] if not matches.empty else clicked_row

        prepared_full = utils.prepare_results_display(full_df_state)
        common_columns = [
            column
            for column in displayed_df.columns
            if column in prepared_full.columns
        ]
        if not common_columns:
            return None

        clicked_values = clicked_row[common_columns].astype(str)

        original_index = clicked_row.name
        if original_index in full_df_state.index and original_index in prepared_full.index:
            candidate = prepared_full.loc[original_index]
            if isinstance(candidate, pd.DataFrame):
                candidate = candidate.iloc[0]
            if candidate[common_columns].astype(str).equals(clicked_values):
                row = full_df_state.loc[original_index]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row

        for idx, candidate in prepared_full[common_columns].iterrows():
            if candidate.astype(str).equals(clicked_values):
                row = full_df_state.loc[idx]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                return row
        return None

    def _fill_row_with_text(self, row_data, entry, tenant, text):
        needs, reason = self._parse_follow_up_fields(text)
        # Вызначаем спасылку на запіс у залежнасці ад пастаўшчыка тэлефаніі.
        if getattr(tenant, "provider", "").lower() == "mts_vats":
            raw = getattr(entry, "raw", {}) or {}
            record_url = str(raw.get("record") or "").strip()
            # Калі ў сырых дадзеных ёсць прамы URL запісу (МТС VATS),
            # выкарыстоўваем яго; інакш падаем на recording_url().
            link = record_url or tenant.recording_url(entry.unique_id)
        else:
            raw = getattr(entry, "raw", {}) or {}
            link = str(raw.get("recording_url") or "").strip()
            if not link:
                link = tenant.recording_url(entry.unique_id)
        row_data["Needs follow-up"] = needs
        row_data["Reason"] = reason
        row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>' if link else ""
        row_data["Status"] = "✅"

    @staticmethod
    def _fill_row_error(row_data, reason: str):
        row_data["Needs follow-up"] = ""
        row_data["Reason"] = reason
        row_data["Link"] = ""
        row_data["Status"] = "❌"

    def _should_use_vertex_batch(self) -> bool:
        if not self.deps.batch_params.enable_gemini_batch:
            return False
        if self.deps.batch_params.batch_mode != "vertex_batch":
            return False
        if not self.deps.project_imports_available:
            return False
        if not self.deps.batch_model_key:
            return False
        try:
            provider = self.deps.ai_registry.get(self.deps.batch_model_key)
        except Exception:
            return False
        return getattr(provider, "provider_name", "") == "gemini"

    @staticmethod
    def _empty_report_result(message: str):
        empty = pd.DataFrame()
        return message, empty, empty, empty, empty

    @staticmethod
    def _format_usage_summary(summary: dict[str, object]) -> str:
        currency = "USD"
        return (
            "### Usage summary\n\n"
            f"- Calls: {summary['total_calls']}\n"
            f"- Duration: {summary['total_duration_seconds']}s / "
            f"{summary['total_duration_minutes']} min\n"
            f"- Tokens: {summary['total_tokens']} total "
            f"(prompt {summary['prompt_tokens']}, output {summary['output_tokens']})\n"
            f"- Internal cost: {summary['estimated_cost']} {currency}\n"
            f"- Client price: {summary['estimated_client_price']} {currency}\n"
            f"- Margin: {summary['margin']} {currency}"
        )

    def has_auth_users(self) -> bool:
        auth_service = getattr(self.deps, "auth_service", None)
        if auth_service is None or not callable(getattr(auth_service, "authenticate", None)):
            return False

        repository = getattr(auth_service, "_repository", None)
        users_by_id = getattr(repository, "_users_by_id", None)
        if users_by_id is not None:
            return bool(users_by_id)

        return True

    @staticmethod
    def _tenant_summary_dict(tenant) -> dict[str, str]:
        tenant_id = str(getattr(tenant, "tenant_id", ""))
        return {
            "tenant_id": tenant_id,
            "display_name": str(getattr(tenant, "display_name", "") or tenant_id),
            "role": str(getattr(tenant, "role", "")),
        }

    @classmethod
    def _auth_session(cls, user, allowed_tenants) -> dict[str, object]:
        return {
            "authenticated": True,
            "user_id": str(getattr(user, "user_id", "")),
            "login": str(getattr(user, "login", "")),
            "allowed_tenants": [
                cls._tenant_summary_dict(tenant) for tenant in allowed_tenants
            ],
        }

    @staticmethod
    def _tenant_dropdown_update(allowed_tenants):
        choices = [
            (
                f"{tenant['display_name']} ({tenant['role']})"
                if tenant.get("role")
                else tenant["display_name"],
                tenant["tenant_id"],
            )
            for tenant in allowed_tenants
        ]
        selected = allowed_tenants[0]["tenant_id"] if len(allowed_tenants) == 1 else None
        return gr.update(choices=choices, value=selected, visible=True)

    @staticmethod
    def _empty_tenant_dropdown_update():
        return gr.update(choices=[], value=None, visible=True)

    @staticmethod
    def _admin_tenants(allowed_tenants):
        return [
            tenant
            for tenant in allowed_tenants
            if str(tenant.get("role", "")).strip().casefold() == "admin"
        ]

    def _auth_session_active(self, auth_session) -> bool:
        if not self.has_auth_users():
            return False
        if not isinstance(auth_session, dict):
            return False
        return bool(auth_session.get("authenticated") and auth_session.get("user_id"))

    def _default_handler_authed(self) -> bool:
        if self.has_auth_users():
            return False
        return os.environ.get("VOCHI_UI_PASSWORD", "") == ""

    def _normalize_auth_args(self, authed=None, auth_session=None):
        if auth_session is None and isinstance(authed, dict):
            return True, authed
        if authed is None:
            authed = self._default_handler_authed()
        return bool(authed), auth_session

    @staticmethod
    def _allowed_tenant_ids(auth_session) -> set[str]:
        if not isinstance(auth_session, dict):
            return set()
        tenants = auth_session.get("allowed_tenants") or []
        return {
            str(tenant.get("tenant_id"))
            for tenant in tenants
            if isinstance(tenant, dict) and tenant.get("tenant_id")
        }

    def _authorize_tenant(self, tenant_id, authed=True, auth_session=None):
        if self.has_auth_users():
            # Auth-service mode always requires a live authenticated session.
            if not self._auth_session_active(auth_session):
                return False, None, "Access denied. Sign in to continue."

            selected_tenant = (tenant_id or "").strip()
            allowed_ids = self._allowed_tenant_ids(auth_session)
            if not selected_tenant and len(allowed_ids) == 1:
                selected_tenant = next(iter(allowed_ids))
            if selected_tenant not in allowed_ids:
                return False, None, "Access denied for the selected tenant."

            auth_service = getattr(self.deps, "auth_service", None)
            user_id = str(auth_session.get("user_id"))
            try:
                if not auth_service.can_access_tenant(user_id, selected_tenant):
                    return False, None, "Access denied for the selected tenant."
            except Exception:
                return False, None, "Access denied for the selected tenant."
            return True, selected_tenant, ""

        if not authed:
            return False, None, "Enter the password to continue."

        return True, (tenant_id or config.DEFAULT_TENANT_ID).strip(), ""

    @staticmethod
    def _legacy_password_result(pwd: str):
        ui_password = os.environ.get("VOCHI_UI_PASSWORD", "")

        if not ui_password:
            return (
                False,
                "âš ï¸ <b>VOCHI_UI_PASSWORD</b> is not configured. Access granted without password.",
                gr.update(visible=False),
            )

        if (pwd or "").strip() == ui_password:
            return True, "âœ… Access granted.", gr.update(visible=False)

        return False, "âŒ Incorrect password.", gr.update(visible=True)

    # ----------------------------------------------------------------------------
    # Usage report handlers
    # ----------------------------------------------------------------------------
    def load_usage_report(
        self,
        tenant_id,
        date_from,
        date_to,
        mode,
        model_key,
        call_user,
        authed,
        auth_session=None,
    ):
        if not authed:
            return self._empty_report_result("🔐 Enter the password to load usage reports.")

        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            return self._empty_report_result(denial)

        repository = getattr(self.deps, "usage_report_repository", None)
        if repository is None:
            return self._empty_report_result(
                "Usage reporting is not configured. Set SUPABASE_URL and SUPABASE_KEY."
            )

        try:
            filters = UsageReportFilters(
                tenant_id=(selected_tenant or "").strip() or None,
                date_from=(date_from or "").strip() or None,
                date_to=(date_to or "").strip() or None,
                mode=mode or ALL_VALUE,
                model_key=model_key or ALL_VALUE,
                call_user=call_user or ALL_VALUE,
            )
            report = build_usage_report(repository.list_usage(filters))
            return (
                self._format_usage_summary(report.summary),
                report.by_model_mode,
                report.by_user,
                report.details,
                report.details,
            )
        except Exception as exc:
            return self._empty_report_result(f"Usage report failed: {exc}")

    def load_usage_report_filter_choices(self, tenant_id, authed, auth_session=None):
        if not authed:
            return (
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                "🔐 Enter the password to refresh report filters.",
            )

        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            return (
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                denial,
            )

        repository = getattr(self.deps, "usage_report_repository", None)
        if repository is None:
            return (
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                "Usage reporting is not configured.",
            )

        try:
            values = repository.list_filter_values((selected_tenant or "").strip() or None)
            return (
                gr.update(choices=values["models"], value=values["models"][0]),
                gr.update(choices=values["modes"], value=values["modes"][0]),
                gr.update(choices=values["users"], value=values["users"][0]),
                "",
            )
        except Exception as exc:
            return (
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                gr.update(choices=[ALL_VALUE], value=ALL_VALUE),
                f"Report filter refresh failed: {exc}",
            )

    @staticmethod
    def export_usage_report(details_df):
        if details_df is None or details_df.empty:
            return gr.update(value=None, visible=False), "❌ No report data to export."

        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            details_df.to_csv(tmp.name, index=False)
            return gr.update(value=tmp.name, visible=True), "✅ File is ready to save."

    def _run_gemini_batch_analysis(self, entries, tenant, prompt_override):
        prompt_text = prompt_override or self.deps.batch_prompt_text or ""
        lang_instruction = GeminiAIAdapter._system_instruction(self.deps.batch_language)
        merged_prompt = f"[SYSTEM INSTRUCTION: {lang_instruction}]\n\n{prompt_text}".strip()

        # Resolve provider info for cache key
        provider = self.deps.ai_registry.get(self.deps.batch_model_key)
        provider_name = getattr(provider, "provider_name", self.deps.batch_model_key)
        prompt_version = self.deps.prompt_service.get_prompt(
            self.deps.batch_prompt_key,
            tenant_id=tenant.tenant_id,
        ).version
        
        # Determine strict prompt fragment for cache key compatibility
        # If prompt matches default config text, we treat custom_fragment as empty
        # to match AnalysisService behavior (which uses prompt_key template).
        custom_fragment = ""
        if (prompt_override or "").strip() != (self.deps.batch_prompt_text or "").strip():
             custom_fragment = (prompt_override or "").strip()

        final_rows = [None] * len(entries)
        tasks: list[BatchTask] = []
        task_indices: list[int] = []

        # Check cache first
        print(f"DEBUG: Checking cache for {len(entries)} entries...")
        for idx, entry in enumerate(entries):
            cache_key = (
                tenant.tenant_id,
                entry.unique_id,
                self.deps.batch_prompt_key,
                prompt_version,
                provider_name,
                self.deps.batch_model_key,
                custom_fragment,
            )
            
            # Access cache directly (it's a MutableMapping)
            cached_result = self.deps.analysis_service._cache.get(cache_key)
            if cached_result:
                print(f"DEBUG: Cache HIT for {entry.unique_id}")
                row_data = self._build_row_base(entry)
                self._fill_row_with_text(row_data, entry, tenant, cached_result.text)
                final_rows[idx] = row_data
            else:
                print(f"DEBUG: Cache MISS for {entry.unique_id}. Key: {cache_key}")
                handle = self.deps.call_log_service.ensure_recording(entry.unique_id, tenant)
                mime_type = guess_mime_type(handle.local_uri)
                tasks.append(
                    BatchTask(
                        key=entry.unique_id,
                        path=handle.local_uri,
                        mime_type=mime_type,
                    )
                )
                task_indices.append(idx)

        # Run batch only for missing tasks
        if tasks:
            runner = VertexBatchRunner(model=self.deps.batch_model_key)
            result_map = runner.run_batch(
                tasks,
                merged_prompt,
                chunk_size=self.deps.batch_params.batch_size,
            )

            for i, task in enumerate(tasks):
                original_idx = task_indices[i]
                entry = entries[original_idx]
                row_data = self._build_row_base(entry)
                
                text_result = result_map.get(entry.unique_id)
                print(f"DEBUG BATCH result for {entry.unique_id}: {text_result!r:.500}")
                if text_result:
                    if text_result.startswith("Error:"):
                        self._fill_row_error(row_data, text_result)
                    else:
                        self._fill_row_with_text(row_data, entry, tenant, text_result)
                        
                        # Save success result to cache
                        # We must replicate logic of AnalysisService.analyze_call key generation
                        cache_key = (
                            tenant.tenant_id,
                            entry.unique_id,
                            self.deps.batch_prompt_key,
                            prompt_version,
                            provider_name,
                            self.deps.batch_model_key,
                            custom_fragment,
                        )
                        new_result = AnalysisResult(
                            text=text_result,
                            model=self.deps.batch_model_key,
                            provider=provider_name,
                            metadata={"batch": True}
                        )
                        # This triggers _save() in FileBackedCache
                        self.deps.analysis_service._cache[cache_key] = new_result

                else:
                    self._fill_row_error(row_data, "No result returned.")
                
                final_rows[original_idx] = row_data

        rows = [r for r in final_rows if r is not None]
        
        ok_count = len([row for row in rows if row.get("Status") == "✅"])
        cached_count = len(entries) - len(tasks)
        processed_count = len(tasks)
        
        final_msg = (
            "✅ Gemini BATCH completed. "
            f"Found: {len(entries)} (Cached: {cached_count}, Processed: {processed_count}), "
            f"Success: {ok_count}"
        )
        return rows, final_msg

    # ----------------------------------------------------------------------------
    # Gradio handlers
    # ----------------------------------------------------------------------------
    def filter_calls(
        self,
        date_value,
        time_from_value,
        time_to_value,
        call_type_value,
        authed,
        tenant_id,
        auth_session=None,
    ):
        """Фільтруе званкі і вяртае табліцу."""
        if not authed:
            return (
                gr.update(value=pd.DataFrame(), visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                "🔐 Enter the password to apply the filter.",
                gr.update(visible=True),
                gr.update(visible=False),
            )

        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            return (
                gr.update(value=pd.DataFrame(), visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                denial,
                gr.update(visible=False),
                gr.update(visible=False),
            )

        if not self.deps.project_imports_available:
            return (
                pd.DataFrame(),
                gr.update(visible=False),
                [],
                "Project dependencies are not loaded.",
                gr.update(visible=False),
                gr.update(visible=False),
            )

        try:
            day = utils.parse_day(date_value)
            time_from = utils.parse_time_value(time_from_value)
            time_to = utils.parse_time_value(time_to_value)
            utils.validate_time_range(time_from, time_to)
            call_type = utils.resolve_call_type(call_type_value)

            tenant = self.deps.tenant_service.resolve(selected_tenant or None)
            entries = self.deps.call_log_service.list_calls(
                day,
                tenant,
                time_from=time_from,
                time_to=time_to,
                call_type=call_type,
            )
            df = pd.DataFrame([entry.raw for entry in entries])
            dd = utils.build_dropdown(df)
            msg = f"Calls found: {len(df)}"

            return (
                gr.update(value=df, visible=True),
                gr.update(visible=False),
                dd,
                msg,
                gr.update(visible=False),
                gr.update(visible=False),
            )
        except Exception as exc:
            return (
                gr.update(value=pd.DataFrame(), visible=True),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                f"Load error: {exc}",
                gr.update(visible=False),
                gr.update(visible=False),
            )

    def play_audio(
        self,
        selected_idx,
        df,
        tenant_id,
        authed,
        auth_session=None,
        current_uid=None,
    ):
        """Прайграць аўдыё па выбраным радку."""
        if not self.deps.project_imports_available:
            return "Project dependencies are not loaded.", None, ""

        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            return denial, None, ""

        unique_id = None
        row = None
        current_uid_value = str(current_uid or "").strip()

        if selected_idx is not None:
            try:
                selected_value = str(selected_idx).strip()
                # Калі карыстальнік перадаў прамы UID радком
                if (
                    current_uid_value
                    and not selected_value.isdigit()
                    and (selected_value.startswith("Batch:") or "|" in selected_value)
                ):
                    unique_id = current_uid_value
                elif not selected_value.isdigit():
                    unique_id = selected_value
                # Калі выбар – індэкс радка ў табліцы
                elif df is not None and not df.empty:
                    row = df.iloc[int(selected_value)]
                    # Спачатку спрабуем стандартнае поле UniqueId (VoChi, батч-вынікі)
                    value = row.get("UniqueId")
                    # Fallback для іншых API (напрыклад, МТС VATS, дзе ёсць uid)
                    if not value:
                        for key in ("uid", "Uid", "UID", "unique_id", "UniqueID", "id", "Id", "ID"):
                            if key in row and row.get(key):
                                value = row.get(key)
                                break
                    unique_id = str(value or "").strip()
            except (ValueError, IndexError):
                return "<em>Invalid selection.</em>", None, ""
        elif current_uid_value:
            unique_id = current_uid_value

        # Адфільтраваць выпадкі, калі UID фактычна не зададзены
        if not unique_id or str(unique_id).strip().lower() in {"none", "nan"}:
            return "<em>Select a call to play.</em>", None, ""

        try:
            tenant = self.deps.tenant_service.resolve(selected_tenant or None)
            handle = self.deps.call_log_service.ensure_recording(unique_id, tenant)

            # Для VoChi выкарыстоўваем стандартны URL, для МТС аддаем перавагу
            # фактычнаму URL запісу з тэлефоннага адаптара (handle.source_uri).
            listen_url = handle.source_uri or tenant.recording_url(unique_id)
            html = f'URL: <a id="audio-listen-link" href="{listen_url}">{listen_url}</a>'

            return html, handle.local_uri, "Ready ✅"
        except Exception as exc:
            return f"Playback failed: {exc}", None, ""

    @staticmethod
    def toggle_custom_prompt(template_key):
        """Паказаць/схаваць поле Custom prompt."""
        return gr.update(visible=(template_key == "custom"))

    def mass_analyze(
        self,
        date_value,
        time_from_value,
        time_to_value,
        call_type_value,
        tenant_id,
        authed,
        auth_session=None,
    ):
        """
        Масавы аналіз (STREAMING).
        Гэта генератар (yield), Gradio будзе адлюстроўваць вынікі паступова.
        Паведамленні прагрэс-статусу і выніковае паведамленне ідуць буйным шрыфтам (Markdown ## / ###).
        """

        yield from self._run_mass_analyze(
            date_value,
            time_from_value,
            time_to_value,
            call_type_value,
            tenant_id,
            authed,
            auth_session,
            custom_prompt_override=None,
        )

    def build_custom_batch_prompt(self, conditions_text: str) -> str:
        base_template = self.deps.batch_custom_prompt_template or "{CONDITIONS}"
        clean_conditions = (conditions_text or "").strip()
        return base_template.replace("{{CONDITIONS}}", clean_conditions)

    def render_custom_prompt(self, conditions_text: str) -> str:
        """Build full prompt text for preview and execution."""
        return self.build_custom_batch_prompt(conditions_text)

    def open_custom_prompt(self, stored_conditions: str):
        """Паказаць акно з умовамі для кастомнага батча."""
        clean_value = (stored_conditions or "").strip()
        return (
            gr.update(value=clean_value),
            gr.update(visible=True),
        )

    @staticmethod
    def close_custom_prompt():
        """Схаваць акно з умовамі."""
        return gr.update(visible=False)

    @staticmethod
    def save_custom_conditions(conditions_text: str):
        """Захаваць умовы і закрыць акно перад стартам."""
        return (conditions_text or "").strip(), gr.update(visible=False)

    def filter_batch_results(self, filter_option, full_df):
        """Фільтрацыя вынікаў батча."""
        if full_df is None or full_df.empty:
            return pd.DataFrame()

        if filter_option == "Needs follow-up":
            filtered = full_df[full_df["Needs follow-up"] == "Yes"]
        elif filter_option == "No follow-up":
            filtered = full_df[full_df["Needs follow-up"] == "No"]
        else:
            filtered = full_df

        return utils.prepare_results_display(filtered)

    def mass_analyze_custom(
        self,
        conditions_text: str,
        date_value,
        time_from_value,
        time_to_value,
        call_type_value,
        tenant_id,
        authed,
        auth_session=None,
    ):
        """Запуск батча з карыстальніцкім промптам."""

        custom_prompt = self.build_custom_batch_prompt(conditions_text)
        yield from self._run_mass_analyze(
            date_value,
            time_from_value,
            time_to_value,
            call_type_value,
            tenant_id,
            authed,
            auth_session,
            custom_prompt_override=custom_prompt,
        )

    def _run_mass_analyze(
        self,
        date_value,
        time_from_value,
        time_to_value,
        call_type_value,
        tenant_id,
        authed,
        auth_session=None,
        *,
        custom_prompt_override: str | None,
    ):
        empty_df = pd.DataFrame()
        hidden_df_update = gr.update(value=empty_df, visible=False)
        empty_state = pd.DataFrame()
        hidden_file = gr.update(value=None, visible=False)
        hidden_filter = gr.update(visible=False)
        visible_filter = gr.update(visible=True)

        def h3(txt: str) -> str:
            return f"### {txt}"

        def h2_success(txt: str) -> str:
            return f"## {txt}"

        def h2_error(txt: str) -> str:
            return f"## {txt}"

        if not authed:
            yield (
                hidden_df_update,
                empty_state,
                h2_error("🔐 Enter the password to run batch analysis."),
                hidden_file,
                hidden_filter,
            )
            return

        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            yield (
                hidden_df_update,
                empty_state,
                h2_error(denial),
                hidden_file,
                hidden_filter,
            )
            return

        if not self.deps.project_imports_available:
            yield (
                hidden_df_update,
                empty_state,
                h2_error("Project dependencies are not loaded."),
                hidden_file,
                hidden_filter,
            )
            return

        if len(self.deps.ai_registry) == 0 or not self.deps.batch_model_key:
            yield (
                hidden_df_update,
                empty_state,
                h2_error("❌ Batch analysis is unavailable: AI model is not configured."),
                hidden_file,
                hidden_filter,
            )
            return

        try:
            day = utils.parse_day(date_value)
            time_from = utils.parse_time_value(time_from_value)
            time_to = utils.parse_time_value(time_to_value)
            utils.validate_time_range(time_from, time_to)
            call_type = utils.resolve_call_type(call_type_value)

            tenant = self.deps.tenant_service.resolve(selected_tenant or None)
            entries = self.deps.call_log_service.list_calls(
                day,
                tenant,
                time_from=time_from,
                time_to=time_to,
                call_type=call_type,
            )

            if not entries:
                yield (
                    hidden_df_update,
                    empty_state,
                    h3("ℹ️ No calls for the selected filter."),
                    hidden_file,
                    hidden_filter,
                )
                return

            total = len(entries)
            settings_service = self.deps.tenant_settings_service
            orchestrator = self.deps.batch_orchestrator
            if settings_service is None or orchestrator is None:
                raise RuntimeError("Batch orchestration is not configured.")
            settings = settings_service.resolve(tenant.tenant_id)
            if settings.batch_model_key not in self.deps.ai_registry:
                raise RuntimeError("Batch analysis is unavailable: AI model is not configured.")
            prompt_override = custom_prompt_override
            # Removed forcing prompt_override to self.deps.batch_prompt_text when None
            # to allow AnalysisService to use default prompt logic (custom_fragment="")
            # which aligns with runner.py/scheduler cache keys.

            yield (
                gr.update(value=pd.DataFrame(), visible=False),
                empty_state,
                h3(f"Starting batch analysis for {total} call(s)..."),
                hidden_file,
                hidden_filter,
            )

            items_by_id = {}
            snapshots = []

            def capture_progress(event):  # noqa: ANN001
                if event.unique_id is None:
                    return
                if event.item is not None:
                    items_by_id[event.unique_id] = event.item
                elif event.unique_id not in items_by_id:
                    return
                rows = [
                    build_batch_item_row(items_by_id[entry.unique_id], tenant)
                    for entry in entries
                    if entry.unique_id in items_by_id
                ]
                snapshots.append((event, pd.DataFrame(rows)))

            run_result = orchestrator.run_with_settings(
                entries,
                tenant,
                settings,
                primary_prompt_key=self.deps.batch_prompt_key,
                primary_custom_prompt=prompt_override or "",
                primary_usage_mode="ui_mass",
                verification_usage_mode="ui_mass_verify",
                progress=capture_progress,
            )

            for event, partial_df in snapshots:
                phase = "Verification" if event.stage_name == "verification" else "Primary analysis"
                interim_msg = f"{phase} {event.completed}/{event.total}... UID `{event.unique_id}`"
                yield (
                    gr.update(value=utils.prepare_results_display(partial_df), visible=True),
                    partial_df,
                    h3(interim_msg),
                    hidden_file,
                    hidden_filter,
                )

            final_df = pd.DataFrame(
                [build_batch_item_row(item, tenant) for item in run_result.items],
            )
            final_msg = (
                "Batch analysis completed. "
                f"total: {run_result.total}, "
                f"round 1 success: {run_result.round_1_success}, "
                f"verification requested: {run_result.verification_requested}, "
                f"verification success: {run_result.verification_success}, "
                f"changed to no follow-up: {run_result.verification_changed_to_false}, "
                f"verification failed: {run_result.verification_failed}, "
                f"final follow-up: {run_result.final_follow_up}"
            )

            yield (
                gr.update(value=utils.prepare_results_display(final_df), visible=True),
                final_df,
                h2_success(final_msg),
                hidden_file,
                visible_filter,
            )
            return

            rows = []  # pragma: no cover - legacy path retained temporarily

            for i, entry in enumerate(entries, start=1):
                pct = int((i / total) * 100)
                row_data = self._build_row_base(entry)

                try:
                    result = self.deps.analysis_service.analyze_call(
                        unique_id=entry.unique_id,
                        tenant=tenant,
                        lang=self.deps.batch_language,
                        options=AnalysisOptions(
                            model_key=self.deps.batch_model_key,
                            prompt_key=self.deps.batch_prompt_key,
                            custom_prompt=prompt_override,
                            mode="ui_mass",
                            call_entry=entry,
                        ),
                    )

                    self._fill_row_with_text(row_data, entry, tenant, result.text)
                except Exception as exc:
                    error_msg = str(exc)
                    # Check if it's a retryable error that failed after all retries
                    if "503" in error_msg or "UNAVAILABLE" in error_msg or "overloaded" in error_msg.lower():
                        self._fill_row_error(
                            row_data,
                            "⏳ Model overloaded (retried 5 times, failed)",
                        )
                    else:
                        self._fill_row_error(row_data, f"❌ {exc}")

                rows.append(row_data)

                partial_df = pd.DataFrame(rows)
                interim_msg = f"Analyzing {i}/{total} ({pct}%)… UID `{entry.unique_id}`"

                yield (
                    gr.update(value=utils.prepare_results_display(partial_df), visible=True),
                    partial_df,
                    h3(interim_msg),
                    hidden_file,
                    hidden_filter,
                )

            final_df = pd.DataFrame(rows)
            ok_count = len(final_df[final_df["Status"] == "✅"])
            final_msg = (
                "✅ Batch analysis completed. "
                f"Found: {total}, processed successfully: {ok_count}"
            )

            yield (
                gr.update(value=utils.prepare_results_display(final_df), visible=True),
                final_df,
                h2_success(final_msg),
                hidden_file,
                visible_filter,
            )

        except Exception as exc:
            yield (
                hidden_df_update,
                empty_state,
                h2_error(f"❌ Analysis failed: {exc}"),
                hidden_file,
                hidden_filter,
            )
            return

    @staticmethod
    def hide_call_list():
        """Схаваць ручны спіс выклікаў пасля батча."""
        return gr.update(visible=False)

    @staticmethod
    def export_results(results_df):
        """Захаваць батч-аналіз у CSV і вярнуць файл у UI."""
        if results_df is None or results_df.empty:
            return gr.update(value=None, visible=False), "❌ No data to export."

        with tempfile.NamedTemporaryFile(
            "w", suffix=".csv", delete=False, encoding="utf-8"
        ) as tmp:
            results_df.to_csv(tmp.name, index=False)
            return gr.update(value=tmp.name, visible=True), "✅ File is ready to save."

    def send_results_email(
        self,
        results_df,
        filter_option: str,
        report_date,
        tenant_id: str,
        authed: bool,
        auth_session=None,
    ) -> str:
        """Send the complete CSV and a filtered HTML table by email."""
        if not authed:
            return "🔐 Enter the password to send email."
        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            return denial
        if results_df is None or results_df.empty:
            return "❌ No data to send."
        if self.deps.email_report_service is None:
            return "❌ Email is not configured. Set BREVO_API_KEY or GOOGLE_app."

        recipient = os.environ.get("EMAIL_TO", "").strip() or "tuttstt@gmail.com"
        try:
            day = utils.parse_day(report_date)
            self.deps.email_report_service.send(
                results_df,
                filter_option=filter_option or "All",
                report_date=day.isoformat(),
                tenant_id=(selected_tenant or config.DEFAULT_TENANT_ID).strip(),
            )
            return f"✅ Email sent to {recipient}."
        except Exception as exc:
            return f"❌ Email sending failed: {exc}"

    @staticmethod
    def _legacy_check_password_unused(pwd: str):
        """Праверка доступу ў UI."""
        _UI_PASSWORD = os.environ.get("VOCHI_UI_PASSWORD", "")

        if not _UI_PASSWORD:
            return (
                False,
                "⚠️ <b>VOCHI_UI_PASSWORD</b> is not configured. Access granted without password.",
                gr.update(visible=False),
            )

        if (pwd or "").strip() == _UI_PASSWORD:
            return True, "✅ Access granted.", gr.update(visible=False)

        return False, "❌ Incorrect password.", gr.update(visible=True)

    def check_password(self, login_or_password: str, password: str | None = None):
        """Authenticate either the new login/password form or legacy password gate."""
        full_response = password is not None

        if not self.has_auth_users():
            legacy_password = password if full_response else login_or_password
            authed, message, group_update = self._legacy_password_result(legacy_password)
            if not full_response:
                return authed, message, group_update
            return (
                authed,
                {},
                message,
                group_update,
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=True),
            )

        auth_service = getattr(self.deps, "auth_service", None)
        user = auth_service.authenticate((login_or_password or "").strip(), password or "")
        if user is None:
            empty_update = self._empty_tenant_dropdown_update()
            response = (
                False,
                {},
                "Incorrect login or password.",
                gr.update(visible=True),
                empty_update,
                empty_update,
                gr.update(visible=False),
            )
            if full_response:
                return response
            return response[0], response[2], response[3]

        allowed_tenants = auth_service.list_allowed_tenants(user.user_id)
        session = self._auth_session(user, allowed_tenants)
        tenant_update = self._tenant_dropdown_update(session["allowed_tenants"])
        admin_tenants = self._admin_tenants(session["allowed_tenants"])
        response = (
            True,
            session,
            "Access granted.",
            gr.update(visible=False),
            tenant_update,
            self._tenant_dropdown_update(admin_tenants),
            gr.update(visible=bool(admin_tenants)),
        )
        if full_response:
            return response
        return response[0], response[2], response[3]

    @staticmethod
    def show_current_uid(current_uid: str):
        """Паказаць выбраны UID у табе AI Analysis."""
        uid = (current_uid or "").strip()
        return (
            f"**Selected UniqueId:** `{uid}`"
            if uid
            else "No file selected for AI Analysis."
        )

    def analyze_bridge(
        self,
        selected_idx,
        df,
        batch_df,
        template_key,
        custom_prompt,
        lang_code,
        model_pref,
        tenant_id,
        current_uid,
        authed,
        auth_session=None,
    ):
        """
        Аналіз адной размовы З ПРАГРЭСАМ.
        """

        uid_to_analyze = (current_uid or "").strip()
        row = None
        if not uid_to_analyze and selected_idx is not None and not str(selected_idx).isdigit():
            uid_to_analyze = str(selected_idx).strip()
        if not uid_to_analyze and selected_idx is not None and df is not None and not df.empty:
            try:
                row = df.iloc[int(selected_idx)]
                uid_to_analyze = str(row.get("UniqueId") or "").strip()
            except (ValueError, IndexError):
                uid_to_analyze = ""
        elif selected_idx is not None and df is not None and not df.empty:
            try:
                row = df.iloc[int(selected_idx)]
            except (ValueError, IndexError):
                row = None
        if row is None and uid_to_analyze:
            row = self._find_row_by_unique_id(batch_df, uid_to_analyze)
        if row is None and uid_to_analyze:
            row = self._single_visible_row_without_unique_id(batch_df)

        if not uid_to_analyze:
            yield "Select a call from the list or batch results first."
            return

        allowed, selected_tenant, denial = self._authorize_tenant(
            tenant_id,
            authed,
            auth_session,
        )
        if not allowed:
            yield denial
            return

        if not self.deps.project_imports_available:
            yield "Project dependencies are not loaded."
            return

        if len(self.deps.ai_registry) == 0:
            yield "❌ No AI models are configured."
            return

        if model_pref not in self.deps.ai_registry:
            yield "❌ Selected model is not available."
            return

        yield (
            "### Preparing analysis...\n\n"
            f"- UID: `{uid_to_analyze}`\n- Model: `{model_pref}`\n- Lang: `{lang_code}`\n\n"
            "Please wait…"
        )

        try:
            tenant = self.deps.tenant_service.resolve(selected_tenant or None)
            lang = config.Language(lang_code)

            result = self.deps.analysis_service.analyze_call(
                unique_id=uid_to_analyze,
                tenant=tenant,
                lang=lang,
                options=AnalysisOptions(
                    model_key=model_pref,
                    prompt_key=template_key,
                    custom_prompt=custom_prompt,
                    mode="ui_direct",
                    call_entry=self._entry_from_row(uid_to_analyze, row),
                ),
            )

            yield f"### Analysis result\n\n{result.text}"

        except Exception as exc:
            yield f"Analysis failed: {exc}"

    def on_batch_row_select(
        self,
        displayed_df: pd.DataFrame,
        full_df_state: pd.DataFrame,
        tenant_id: str,
        authed,
        evt: gr.SelectData,
        auth_session=None,
    ):
        """Апрацоўвае выбар радка з табліцы вынікаў (Batch results)."""
        empty_return = (
            gr.update(choices=[], value=None),
            "",
            "No file selected for AI Analysis.",
            "",
            None,
            "Selection error.",
        )

        if (
            evt is None
            or displayed_df is None
            or displayed_df.empty
        ):
            return empty_return
        if full_df_state is None:
            full_df_state = pd.DataFrame()

        try:
            visual_row_index = self._visual_row_index_from_select_event(evt)
            if visual_row_index is None:
                return empty_return
            original_row = self._find_batch_original_row(
                displayed_df,
                full_df_state,
                visual_row_index,
            )
            if original_row is None:
                return empty_return
            row_dict = original_row.to_dict()
            uid = self._uid_from_row(row_dict)
            if not uid:
                return empty_return

            label = (
                f"{row_dict.get('Start','')} | "
                f"{row_dict.get('Caller','')} → "
                f"{row_dict.get('Destination','')} "
                f"({row_dict.get('Duration (s)','')}s)"
            )
            dd_update = gr.update(choices=[(f"Batch: {label}", uid)], value=uid)
            uid_md_update = self.show_current_uid(uid)

            try:
                allowed, selected_tenant, denial = self._authorize_tenant(
                    tenant_id,
                    authed,
                    auth_session,
                )
                if not allowed:
                    return dd_update, uid, uid_md_update, denial, None, ""
                tenant = self.deps.tenant_service.resolve(selected_tenant or None)
                recording_url = self._recording_url_from_row(row_dict)
                if recording_url:
                    handle = self.deps.call_log_service.ensure_recording(
                        uid,
                        tenant,
                        recording_url,
                    )
                else:
                    handle = self.deps.call_log_service.ensure_recording(uid, tenant)
                listen_url = handle.source_uri or tenant.recording_url(uid)
                html = f'URL: <a id="audio-listen-link" href="{listen_url}">{listen_url}</a>'
                audio_uri = handle.local_uri
                status_msg = "Ready ✅"
            except Exception as exc:
                html, audio_uri, status_msg = f"Playback failed: {exc}", None, ""

            return dd_update, uid, uid_md_update, html, audio_uri, status_msg

        except (AttributeError, IndexError, KeyError, TypeError):
            return empty_return

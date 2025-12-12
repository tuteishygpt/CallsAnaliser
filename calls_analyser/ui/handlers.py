"""Gradio event handlers separated from app wiring."""
from __future__ import annotations

import json
import os
import tempfile

import gradio as gr
import pandas as pd

from . import config
from .dependencies import AppDependencies, AnalysisOptions
from . import utils
from calls_analyser.adapters.ai.gemini import GeminiAIAdapter
from calls_analyser.domain.exceptions import AIModelError
from calls_analyser.domain.models import AnalysisResult
from calls_analyser.services.gemini_batch import BatchTask, GeminiBatchRunner, guess_mime_type


class UIHandlers:
    def __init__(self, deps: AppDependencies):
        self.deps = deps

    # ----------------------------------------------------------------------------
    # Internal helpers
    # ----------------------------------------------------------------------------
    @staticmethod
    def _parse_follow_up_fields(text: str) -> tuple[str, str]:
        text_clean = str(text or "").strip()
        try:
            l, r = text_clean.find("{"), text_clean.rfind("}")
            if l != -1 and r != -1 and r > l:
                text_clean = text_clean[l : r + 1]
            payload = json.loads(text_clean)
            needs_follow_up = payload.get("needs_follow_up")
            needs_str = "Yes" if needs_follow_up else "No"
            reason = str(payload.get("reason") or "")
            return needs_str, reason
        except Exception:
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
        return {
            "Start": entry.started_at.isoformat() if entry.started_at else "",
            "Caller": entry.caller_id or "",
            "Destination": entry.destination or "",
            "Duration (s)": entry.duration_seconds,
            "UniqueId": entry.unique_id,
        }

    def _fill_row_with_text(self, row_data, entry, tenant, text):
        needs, reason = self._parse_follow_up_fields(text)
        link = (
            f"{tenant.vochi_base_url.rstrip('/')}/calllogs/"
            f"{tenant.vochi_client_id}/{entry.unique_id}"
        )
        row_data["Needs follow-up"] = needs
        row_data["Reason"] = reason
        row_data["Link"] = f'<a href="{link}" target="_blank">Listen</a>'
        row_data["Status"] = "✅"

    @staticmethod
    def _fill_row_error(row_data, reason: str):
        row_data["Needs follow-up"] = ""
        row_data["Reason"] = reason
        row_data["Link"] = ""
        row_data["Status"] = "❌"

    def _should_use_gemini_batch(self) -> bool:
        if not self.deps.batch_params.enable_gemini_batch:
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

    def _run_gemini_batch_analysis(self, entries, tenant, prompt_override):
        api_key = self.deps.secrets_adapter.get_optional_secret("GOOGLE_API_KEY")
        if not api_key:
            raise AIModelError("GOOGLE_API_KEY is not configured")

        prompt_text = prompt_override or self.deps.batch_prompt_text or ""
        lang_instruction = GeminiAIAdapter._system_instruction(self.deps.batch_language)
        merged_prompt = f"[SYSTEM INSTRUCTION: {lang_instruction}]\n\n{prompt_text}".strip()

        # Resolve provider info for cache key
        provider = self.deps.ai_registry.get(self.deps.batch_model_key)
        provider_name = getattr(provider, "provider_name", self.deps.batch_model_key)
        
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
            runner = GeminiBatchRunner(api_key=api_key, model=self.deps.batch_model_key)
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
    ):
        """Фільтруе званкі і вяртае табліцу."""
        if not authed:
            return (
                gr.update(value=pd.DataFrame(), visible=False),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                "🔐 Enter the password to apply the filter.",
                gr.update(visible=True),
            )

        if not self.deps.project_imports_available:
            return (
                pd.DataFrame(),
                gr.update(visible=False),
                [],
                "Project dependencies are not loaded.",
                gr.update(visible=False),
            )

        try:
            day = utils.parse_day(date_value)
            time_from = utils.parse_time_value(time_from_value)
            time_to = utils.parse_time_value(time_to_value)
            utils.validate_time_range(time_from, time_to)
            call_type = utils.resolve_call_type(call_type_value)

            tenant = self.deps.tenant_service.resolve(tenant_id or None)
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
            )
        except Exception as exc:
            return (
                gr.update(value=pd.DataFrame(), visible=True),
                gr.update(visible=False),
                gr.update(choices=[], value=None),
                f"Load error: {exc}",
                gr.update(visible=False),
            )

    def play_audio(self, selected_idx, df, tenant_id):
        """Прайграць аўдыё па выбраным радку."""
        if not self.deps.project_imports_available:
            return "Project dependencies are not loaded.", None, ""

        unique_id = None

        if selected_idx is not None:
            try:
                if not str(selected_idx).isdigit():
                    unique_id = str(selected_idx)
                elif df is not None and not df.empty:
                    row = df.iloc[int(selected_idx)]
                    unique_id = str(row.get("UniqueId"))
            except (ValueError, IndexError):
                return "<em>Invalid selection.</em>", None, ""

        if not unique_id:
            return "<em>Select a call to play.</em>", None, ""

        try:
            tenant = self.deps.tenant_service.resolve(tenant_id or None)
            handle = self.deps.call_log_service.ensure_recording(unique_id, tenant)

            listen_url = (
                f"{tenant.vochi_base_url.rstrip('/')}/calllogs/"
                f"{tenant.vochi_client_id}/{unique_id}"
            )
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
        preview = self.build_custom_batch_prompt(clean_value)
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

    def mass_analyze_custom(
        self,
        conditions_text: str,
        date_value,
        time_from_value,
        time_to_value,
        call_type_value,
        tenant_id,
        authed,
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
        *,
        custom_prompt_override: str | None,
    ):
        empty_df = pd.DataFrame()
        hidden_df_update = gr.update(value=empty_df, visible=False)
        hidden_file = gr.update(value=None, visible=False)

        def h3(txt: str) -> str:
            return f"### {txt}"

        def h2_success(txt: str) -> str:
            return f"## {txt}"

        def h2_error(txt: str) -> str:
            return f"## {txt}"

        if not authed:
            yield (
                hidden_df_update,
                h2_error("🔐 Enter the password to run batch analysis."),
                hidden_file,
            )
            return

        if not self.deps.project_imports_available:
            yield (
                hidden_df_update,
                h2_error("Project dependencies are not loaded."),
                hidden_file,
            )
            return

        if len(self.deps.ai_registry) == 0 or not self.deps.batch_model_key:
            yield (
                hidden_df_update,
                h2_error("❌ Batch analysis is unavailable: AI model is not configured."),
                hidden_file,
            )
            return

        try:
            day = utils.parse_day(date_value)
            time_from = utils.parse_time_value(time_from_value)
            time_to = utils.parse_time_value(time_to_value)
            utils.validate_time_range(time_from, time_to)
            call_type = utils.resolve_call_type(call_type_value)

            tenant = self.deps.tenant_service.resolve(tenant_id or None)
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
                    h3("ℹ️ No calls for the selected filter."),
                    hidden_file,
                )
                return

            total = len(entries)
            prompt_override = custom_prompt_override
            if prompt_override is None:
                prompt_override = self.deps.batch_prompt_text or None

            yield (
                gr.update(value=pd.DataFrame(), visible=False),
                h3(f"Starting batch analysis for {total} call(s)..."),
                hidden_file,
            )

            if self._should_use_gemini_batch():
                print("DEBUG: Using Gemini BATCH mode (API)")
                try:
                    rows, final_msg = self._run_gemini_batch_analysis(
                        entries, tenant, prompt_override
                    )
                    final_df = pd.DataFrame(rows)
                    yield (
                        gr.update(value=final_df, visible=True),
                        h2_success(final_msg),
                        hidden_file,
                    )
                    return
                except Exception as exc:
                    yield (
                        hidden_df_update,
                        h2_error(f"❌ Gemini BATCH failed: {exc}"),
                        hidden_file,
                    )
                    return
            else:
                print(f"DEBUG: Gemini Batch disabled. enable_gemini_batch={self.deps.batch_params.enable_gemini_batch}, available={self.deps.project_imports_available}, model={self.deps.batch_model_key}")
                # Fallback to serial processing (one by one)

            rows = []

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
                    gr.update(value=partial_df, visible=True),
                    h3(interim_msg),
                    hidden_file,
                )

            final_df = pd.DataFrame(rows)
            ok_count = len(final_df[final_df["Status"] == "✅"])
            final_msg = (
                "✅ Batch analysis completed. "
                f"Found: {total}, processed successfully: {ok_count}"
            )

            yield (
                gr.update(value=final_df, visible=True),
                h2_success(final_msg),
                hidden_file,
            )

        except Exception as exc:
            yield (
                hidden_df_update,
                h2_error(f"❌ Analysis failed: {exc}"),
                hidden_file,
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

    @staticmethod
    def check_password(pwd: str):
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
        template_key,
        custom_prompt,
        lang_code,
        model_pref,
        tenant_id,
        current_uid,
    ):
        """
        Аналіз адной размовы З ПРАГРЭСАМ.
        """

        uid_to_analyze = (current_uid or "").strip()
        if not uid_to_analyze and selected_idx is not None and df is not None and not df.empty:
            try:
                uid_to_analyze = str(df.iloc[int(selected_idx)].get("UniqueId") or "").strip()
            except (ValueError, IndexError):
                uid_to_analyze = ""

        if not uid_to_analyze:
            yield "Select a call from the list or batch results first."
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
            tenant = self.deps.tenant_service.resolve(tenant_id or None)
            lang = config.Language(lang_code)

            result = self.deps.analysis_service.analyze_call(
                unique_id=uid_to_analyze,
                tenant=tenant,
                lang=lang,
                options=AnalysisOptions(
                    model_key=model_pref,
                    prompt_key=template_key,
                    custom_prompt=custom_prompt,
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
        evt: gr.SelectData,
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
            or full_df_state is None
            or full_df_state.empty
        ):
            return empty_return

        try:
            visual_row_index = evt.index[0]
            clicked_row_from_view = displayed_df.iloc[visual_row_index]
            uid = str(clicked_row_from_view.get("UniqueId", "")).strip()
            if not uid:
                return empty_return

            original_row_series = full_df_state[full_df_state["UniqueId"] == uid]
            if original_row_series.empty:
                return empty_return
            original_row = original_row_series.iloc[0]
            row_dict = original_row.to_dict()

            label = (
                f"{row_dict.get('Start','')} | "
                f"{row_dict.get('Caller','')} → "
                f"{row_dict.get('Destination','')} "
                f"({row_dict.get('Duration (s)','')}s)"
            )
            dd_update = gr.update(choices=[(f"Batch: {label}", uid)], value=uid)
            uid_md_update = self.show_current_uid(uid)

            try:
                tenant = self.deps.tenant_service.resolve(tenant_id or None)
                handle = self.deps.call_log_service.ensure_recording(uid, tenant)
                listen_url = (
                    f"{tenant.vochi_base_url.rstrip('/')}/calllogs/"
                    f"{tenant.vochi_client_id}/{uid}"
                )
                html = f'URL: <a id="audio-listen-link" href="{listen_url}">{listen_url}</a>'
                audio_uri = handle.local_uri
                status_msg = "Ready ✅"
            except Exception as exc:
                html, audio_uri, status_msg = f"Playback failed: {exc}", None, ""

            return dd_update, uid, uid_md_update, html, audio_uri, status_msg

        except (AttributeError, IndexError, KeyError):
            return empty_return

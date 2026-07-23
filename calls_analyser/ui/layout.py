"""UI layout and event wiring for the Gradio app."""
from __future__ import annotations

import os

import gradio as gr
import pandas as pd

from . import config, utils
from .dependencies import AppDependencies
from .handlers import UIHandlers

JS_FIX_LINKS = """
() => {
  setTimeout(() => {
    const links = document.querySelectorAll('a.new-tab-link, a#audio-listen-link');
    links.forEach(link => {
      if (link) {
        link.target = '_blank';
        link.rel = 'noopener noreferrer';
      }
    });
  }, 100);
}
"""


def build_demo(deps: AppDependencies, handlers: UIHandlers) -> gr.Blocks:
    auth_mode = handlers.has_auth_users()
    legacy_password_required = os.environ.get("VOCHI_UI_PASSWORD", "") != ""
    initial_authed = False if auth_mode else not legacy_password_required

    with gr.Blocks(title="Calls Call Logs (Gradio)") as demo:
        gr.Markdown(
            "# Calls→ audio → AI analysis\n"
            "*Filter calls by date, time and type, listen to recordings and run batch AI analysis.*"
        )

        authed = gr.State(initial_authed)
        auth_session = gr.State({})
        batch_results_state = gr.State(pd.DataFrame())
        report_details_state = gr.State(pd.DataFrame())
        current_uid_state = gr.State("")
        custom_batch_conditions_state = gr.State(deps.batch_custom_conditions)

        with gr.Group(visible=(auth_mode or legacy_password_required)) as pwd_group:
            gr.Markdown("### 🔐 Enter password")
            login_tb = gr.Textbox(
                label="Login",
                placeholder="Login",
                lines=1,
                visible=auth_mode,
            )
            pwd_tb = gr.Textbox(
                label="Password", type="password", placeholder="••••••••", lines=1
            )
            pwd_btn = gr.Button("Unlock", variant="primary")
            login_status = gr.Markdown()

        with gr.Tabs() as tabs:  # noqa: F841
            with gr.Tab("Calls"):
                with gr.Row():
                    tenant_tb = gr.Textbox(
                        label="Tenant ID",
                        value=config.DEFAULT_TENANT_ID,
                        scale=1,
                        visible=not auth_mode,
                    )
                    tenant_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Tenant",
                        type="value",
                        allow_custom_value=False,
                        scale=1,
                        visible=auth_mode,
                    )
                    date_inp = gr.Textbox(
                        label="Date", value=utils.yesterday_str(), placeholder="YYYY-MM-DD", scale=1
                    )
                    time_from_inp = gr.Textbox(
                        label="Time from", value="08:00", placeholder="HH:MM", scale=1
                    )
                    time_to_inp = gr.Textbox(label="Time to", value="20:00", placeholder="HH:MM", scale=1)
                    call_type_dd = gr.Dropdown(
                        choices=config.CALL_TYPE_OPTIONS,
                        value="0",
                        label="Call type",
                        type="value",
                        scale=1,
                    )
                with gr.Row():
                    filter_btn = gr.Button("Filter", variant="primary", scale=0)
                    batch_btn = gr.Button("Batch analyze", variant="secondary", scale=0)
                    batch_custom_btn = gr.Button(
                        "Batch analyze custom",
                        variant="secondary",
                        scale=0,
                        # Пакажам кнопку толькі калі BATCH_CUSTOM == "on"
                        visible=(getattr(config, "BATCH_CUSTOM", "") == "on"),
                    )
                    save_btn = gr.Button("Save to file", scale=0)
                    email_btn = gr.Button("Send by email", scale=0)

                with gr.Group(visible=False) as custom_prompt_modal:
                    gr.Markdown("### Batch analyze custom prompt")
                    batch_custom_prompt_tb = gr.Textbox(
                        label="Rules and conditions (editable part)",
                        lines=8,
                        value=deps.batch_custom_conditions,
                    )

                    with gr.Row():
                        custom_start_btn = gr.Button("Start", variant="primary")
                        custom_cancel_btn = gr.Button("Cancel")

                status_fetch = gr.Markdown()
                batch_status_md = gr.Markdown()

                with gr.Row(visible=False) as batch_filter_row:
                    filter_radio = gr.Radio(
                        choices=["All", "Needs follow-up", "No follow-up"],
                        value="Needs follow-up",
                        label="Filter results",
                        interactive=True,
                    )

                calls_df = gr.DataFrame(
                    value=pd.DataFrame(),
                    label="Call list (manual filter)",
                    interactive=False,
                )

                batch_results_df = gr.DataFrame(
                    value=pd.DataFrame(),
                    label="Batch results",
                    interactive=True,
                    visible=False,
                    datatype=[
                        "str",
                        "str",
                        "str",
                        "number",
                        "str",
                        "str",
                        "str",
                        "markdown",
                        "str",
                    ],
                )

                row_dd = gr.Dropdown(
                    choices=[],
                    label="Call",
                    info="Choose a row to listen/analyze",
                    type="value",
                    allow_custom_value=True,
                )

                with gr.Row():
                    play_btn = gr.Button("🎧 Play")

                url_html = gr.HTML()
                audio_out = gr.Audio(label="Audio", type="filepath")
                batch_file = gr.File(label="Export CSV", visible=False)

            with gr.Tab("AI Analysis"):
                with gr.Row():
                    tpl_dd = gr.Dropdown(
                        choices=config.TPL_OPTIONS,
                        value="simple" if config.TPL_OPTIONS else "custom",
                        label="Template",
                    )
                    lang_dd = gr.Dropdown(
                        choices=config.LANG_OPTIONS,
                        value=config.Language.AUTO,
                        label="Language",
                    )
                    model_dd = gr.Dropdown(
                        choices=deps.model_choices,
                        value=deps.model_default,
                        label="Model",
                        interactive=bool(deps.model_options),
                        info=deps.model_info,
                    )

                custom_prompt_tb = gr.Textbox(
                    label="Custom prompt", lines=8, visible=False
                )

                current_uid_md = gr.Markdown(
                    value="No file selected for AI Analysis."
                )

                analyze_btn = gr.Button("🧠 Analyze", variant="primary")
                analysis_md = gr.Markdown()

            with gr.Tab("Tenant Settings", visible=False) as tenant_admin_tab:
                with gr.Row():
                    tenant_admin_dd = gr.Dropdown(
                        choices=[], value=None, label="Administered tenant",
                        allow_custom_value=False, scale=1,
                    )
                    tenant_admin_reload_btn = gr.Button("Reload", scale=0, interactive=False)
                    tenant_admin_save_btn = gr.Button(
                        "Save", variant="primary", scale=0, interactive=False
                    )
                tenant_admin_status = gr.Markdown()
                with gr.Group():
                    gr.Markdown("### General")
                    tenant_admin_id = gr.Textbox(label="Tenant ID", interactive=False)
                    tenant_admin_name = gr.Textbox(label="Display name", interactive=False)
                    tenant_admin_status_dd = gr.Dropdown(
                        choices=["active", "inactive"], value=None, label="Status",
                        interactive=False,
                    )
                with gr.Group():
                    gr.Markdown("### Telephony")
                    tenant_admin_provider = gr.Dropdown(
                        choices=["vochi", "mts_vats"], value=None, label="Provider",
                        interactive=False,
                    )
                    tenant_admin_vochi_url = gr.Textbox(label="VoChi base URL", interactive=False)
                    tenant_admin_vochi_key = gr.Textbox(
                        label="VoChi API key", type="password", interactive=False
                    )
                    tenant_admin_mts_domain = gr.Textbox(
                        label="MTS domain", type="password", interactive=False
                    )
                    tenant_admin_mts_key = gr.Textbox(
                        label="MTS API key", type="password", interactive=False
                    )
                with gr.Group():
                    gr.Markdown("### AI defaults")
                    tenant_admin_default_language = gr.Textbox(
                        label="Default language", interactive=False
                    )
                    tenant_admin_default_model = gr.Textbox(
                        label="Default model key", interactive=False
                    )
                    tenant_admin_batch_language = gr.Textbox(
                        label="Batch language", interactive=False
                    )
                    tenant_admin_batch_model = gr.Textbox(
                        label="Batch model key", interactive=False
                    )
                with gr.Group():
                    gr.Markdown("### Batch processing")
                    tenant_admin_batch_enabled = gr.Dropdown(
                        choices=[("Enabled", True), ("Disabled", False)],
                        label="Batch enabled", value=None, interactive=False,
                    )
                    tenant_admin_batch_size = gr.Number(
                        label="Batch size", value=None, precision=0, interactive=False
                    )
                    tenant_admin_custom_batch = gr.Dropdown(
                        choices=[("Enabled", True), ("Disabled", False)],
                        label="Custom batch enabled", value=None, interactive=False,
                    )
                with gr.Group():
                    gr.Markdown("### Scheduler")
                    tenant_admin_scheduler_enabled = gr.Dropdown(
                        choices=[("Enabled", True), ("Disabled", False)],
                        label="Scheduler enabled", value=None, interactive=False,
                    )
                    tenant_admin_scheduler_mode = gr.Dropdown(
                        choices=["cron", "interval"], value=None, label="Mode",
                        interactive=False,
                    )
                    tenant_admin_cron_time = gr.Textbox(label="Cron time", interactive=False)
                    tenant_admin_interval = gr.Number(
                        label="Interval minutes", value=None, precision=0, interactive=False
                    )
                    tenant_admin_time_from = gr.Textbox(label="Call time from", interactive=False)
                    tenant_admin_time_to = gr.Textbox(label="Call time to", interactive=False)
                    tenant_admin_call_type = gr.Textbox(
                        label="Call type filter", interactive=False
                    )
                with gr.Group():
                    gr.Markdown("### Email")
                    tenant_admin_email_to = gr.Textbox(label="Recipient", interactive=False)
                    tenant_admin_email_from = gr.Textbox(
                        label="Sender address", interactive=False
                    )
                    tenant_admin_email_name = gr.Textbox(label="Sender name", interactive=False)
                with gr.Group():
                    gr.Markdown("### Prompt templates")
                    tenant_admin_prompts = gr.DataFrame(
                        headers=["Key", "Title", "Body", "Version"],
                        datatype=["str", "str", "str", "number"],
                        value=[],
                        row_count=0,
                        row_limits=(0, 0),
                        column_count=4,
                        column_limits=(4, 4),
                        interactive=False,
                    )

            with gr.Tab("Reports", visible=not auth_mode) as reports_tab:
                with gr.Row():
                    tenant_report_tb = gr.Textbox(
                        label="Tenant ID",
                        value=config.DEFAULT_TENANT_ID,
                        scale=1,
                        visible=not auth_mode,
                    )
                    tenant_report_dd = gr.Dropdown(
                        choices=[],
                        value=None,
                        label="Tenant",
                        type="value",
                        allow_custom_value=False,
                        scale=1,
                        visible=auth_mode,
                    )
                    report_date_from_tb = gr.Textbox(
                        label="Date from", placeholder="YYYY-MM-DD", scale=1
                    )
                    report_date_to_tb = gr.Textbox(
                        label="Date to", placeholder="YYYY-MM-DD", scale=1
                    )
                    report_mode_dd = gr.Dropdown(
                        choices=["All", "ui_direct", "ui_mass", "scheduler_batch", "test"],
                        value="All",
                        label="Mode",
                        type="value",
                        scale=1,
                    )
                    report_model_dd = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="Model",
                        type="value",
                        allow_custom_value=True,
                        scale=1,
                    )
                    report_user_dd = gr.Dropdown(
                        choices=["All"],
                        value="All",
                        label="User/operator",
                        type="value",
                        allow_custom_value=True,
                        scale=1,
                    )

                with gr.Row():
                    refresh_report_filters_btn = gr.Button("Refresh filters", scale=0)
                    load_report_btn = gr.Button("Load report", variant="primary", scale=0)
                    report_export_btn = gr.Button("Export CSV", scale=0)

                report_summary_md = gr.Markdown()
                report_status_md = gr.Markdown()
                report_file = gr.File(label="Usage report CSV", visible=False)

                report_model_mode_df = gr.DataFrame(
                    value=pd.DataFrame(),
                    label="By model and mode",
                    interactive=False,
                )
                report_user_df = gr.DataFrame(
                    value=pd.DataFrame(),
                    label="By user/operator",
                    interactive=False,
                )
                report_details_df = gr.DataFrame(
                    value=pd.DataFrame(),
                    label="Usage details",
                    interactive=False,
                )

        tenant_selector = tenant_dd if auth_mode else tenant_tb
        tenant_report_selector = tenant_report_dd if auth_mode else tenant_report_tb

        login_event = pwd_btn.click(
            handlers.check_password,
            inputs=[login_tb, pwd_tb],
            outputs=[
                authed,
                auth_session,
                login_status,
                pwd_group,
                tenant_dd,
                tenant_report_dd,
                reports_tab,
            ],
        )
        tenant_admin_outputs = [
            tenant_admin_id, tenant_admin_name, tenant_admin_status_dd, tenant_admin_provider,
            tenant_admin_vochi_url, tenant_admin_vochi_key, tenant_admin_mts_domain,
            tenant_admin_mts_key, tenant_admin_default_language, tenant_admin_default_model,
            tenant_admin_batch_language, tenant_admin_batch_model, tenant_admin_batch_enabled,
            tenant_admin_batch_size, tenant_admin_custom_batch, tenant_admin_scheduler_enabled,
            tenant_admin_scheduler_mode, tenant_admin_cron_time, tenant_admin_interval,
            tenant_admin_time_from, tenant_admin_time_to, tenant_admin_call_type,
            tenant_admin_email_to, tenant_admin_email_from, tenant_admin_email_name,
            tenant_admin_prompts, tenant_admin_status,
            tenant_admin_reload_btn, tenant_admin_save_btn,
        ]
        tenant_admin_inputs = [
            tenant_admin_name, tenant_admin_status_dd, tenant_admin_provider,
            tenant_admin_vochi_url, tenant_admin_vochi_key, tenant_admin_mts_domain,
            tenant_admin_mts_key, tenant_admin_default_language, tenant_admin_default_model,
            tenant_admin_batch_language, tenant_admin_batch_model, tenant_admin_batch_enabled,
            tenant_admin_batch_size, tenant_admin_custom_batch, tenant_admin_scheduler_enabled,
            tenant_admin_scheduler_mode, tenant_admin_cron_time, tenant_admin_interval,
            tenant_admin_time_from, tenant_admin_time_to, tenant_admin_call_type,
            tenant_admin_email_to, tenant_admin_email_from, tenant_admin_email_name,
        ]
        login_event.then(
            handlers.refresh_admin_tenants,
            inputs=[auth_session],
            outputs=[tenant_admin_dd, tenant_admin_tab],
        ).then(
            handlers.load_tenant_admin_form,
            inputs=[tenant_admin_dd, auth_session],
            outputs=tenant_admin_outputs,
        )
        tenant_admin_change_event = tenant_admin_dd.input(
            handlers.refresh_admin_tenants,
            inputs=[auth_session, tenant_admin_dd],
            outputs=[tenant_admin_dd, tenant_admin_tab],
        )
        tenant_admin_change_event.then(
            handlers.load_tenant_admin_form,
            inputs=[tenant_admin_dd, auth_session],
            outputs=tenant_admin_outputs,
        )

        tenant_admin_reload_event = tenant_admin_reload_btn.click(
            handlers.refresh_admin_tenants,
            inputs=[auth_session, tenant_admin_dd],
            outputs=[tenant_admin_dd, tenant_admin_tab],
        )
        tenant_admin_reload_event.then(
            handlers.load_tenant_admin_form,
            inputs=[tenant_admin_dd, auth_session],
            outputs=tenant_admin_outputs,
        )

        tenant_admin_save_event = tenant_admin_save_btn.click(
            handlers.save_tenant_admin_form,
            inputs=[tenant_admin_dd, *tenant_admin_inputs, auth_session],
            outputs=tenant_admin_outputs,
        )
        tenant_admin_save_event.then(
            handlers.refresh_admin_tenants,
            inputs=[auth_session, tenant_admin_dd],
            outputs=[tenant_admin_dd, tenant_admin_tab],
        )

        filter_btn.click(
            handlers.filter_calls,
            inputs=[
                date_inp,
                time_from_inp,
                time_to_inp,
                call_type_dd,
                authed,
                tenant_selector,
                auth_session,
            ],
            outputs=[calls_df, batch_results_df, row_dd, status_fetch, pwd_group, batch_filter_row],
        )

        batch_btn.click(
            fn=handlers.hide_batch_filter,
            outputs=[batch_filter_row],
        ).then(
            fn=handlers.mass_analyze,
            inputs=[
                date_inp,
                time_from_inp,
                time_to_inp,
                call_type_dd,
                tenant_selector,
                authed,
                auth_session,
            ],
            outputs=[
                batch_results_df,
                batch_results_state,
                batch_status_md,
                batch_file,
            ],
        ).then(
            fn=handlers.filter_batch_results,
            inputs=[filter_radio, batch_results_state],
            outputs=[batch_results_df],
        ).then(
            fn=handlers.update_batch_filter_visibility,
            inputs=[batch_results_state],
            outputs=[batch_filter_row],
        ).then(
            fn=handlers.hide_call_list,
            outputs=[calls_df],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        batch_custom_btn.click(
            handlers.open_custom_prompt,
            inputs=[custom_batch_conditions_state],
            outputs=[batch_custom_prompt_tb, custom_prompt_modal],
        )



        custom_cancel_btn.click(
            handlers.close_custom_prompt,
            outputs=[custom_prompt_modal],
        )

        custom_start_btn.click(
            handlers.save_custom_conditions,
            inputs=[batch_custom_prompt_tb],
            outputs=[custom_batch_conditions_state, custom_prompt_modal],
        ).then(
            fn=handlers.hide_batch_filter,
            outputs=[batch_filter_row],
        ).then(
            fn=handlers.mass_analyze_custom,
            inputs=[
                custom_batch_conditions_state,
                date_inp,
                time_from_inp,
                time_to_inp,
                call_type_dd,
                tenant_selector,
                authed,
                auth_session,
            ],
            outputs=[
                batch_results_df,
                batch_results_state,
                batch_status_md,
                batch_file,
            ],
        ).then(
            fn=handlers.filter_batch_results,
            inputs=[filter_radio, batch_results_state],
            outputs=[batch_results_df],
        ).then(
            fn=handlers.update_batch_filter_visibility,
            inputs=[batch_results_state],
            outputs=[batch_filter_row],
        ).then(
            fn=handlers.hide_call_list,
            outputs=[calls_df],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        batch_results_df.select(
            fn=handlers.on_batch_row_select,
            inputs=[batch_results_df, batch_results_state, tenant_selector, authed, auth_session],
            outputs=[row_dd, current_uid_state, current_uid_md, url_html, audio_out, status_fetch],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        filter_radio.change(
            fn=handlers.filter_batch_results,
            inputs=[filter_radio, batch_results_state],
            outputs=[batch_results_df],
        )

        play_btn.click(
            handlers.play_audio,
            inputs=[row_dd, calls_df, tenant_selector, authed, auth_session, current_uid_state],
            outputs=[url_html, audio_out, status_fetch],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        save_btn.click(
            handlers.export_results,
            inputs=[batch_results_state],
            outputs=[batch_file, batch_status_md],
        )

        email_btn.click(
            handlers.send_results_email,
            inputs=[
                batch_results_state,
                filter_radio,
                date_inp,
                tenant_selector,
                authed,
                auth_session,
            ],
            outputs=[batch_status_md],
        )

        refresh_report_filters_btn.click(
            handlers.load_usage_report_filter_choices,
            inputs=[tenant_report_selector, authed, auth_session],
            outputs=[report_model_dd, report_mode_dd, report_user_dd, report_status_md],
        )

        load_report_btn.click(
            handlers.load_usage_report,
            inputs=[
                tenant_report_selector,
                report_date_from_tb,
                report_date_to_tb,
                report_mode_dd,
                report_model_dd,
                report_user_dd,
                authed,
                auth_session,
            ],
            outputs=[
                report_summary_md,
                report_model_mode_df,
                report_user_df,
                report_details_df,
                report_details_state,
            ],
        )

        report_export_btn.click(
            handlers.export_usage_report,
            inputs=[report_details_state],
            outputs=[report_file, report_status_md],
        )

        tpl_dd.change(
            handlers.toggle_custom_prompt,
            inputs=[tpl_dd],
            outputs=[custom_prompt_tb],
        )

        analyze_btn.click(
            fn=handlers.analyze_bridge,
            inputs=[
                row_dd,
                calls_df,
                batch_results_state,
                tpl_dd,
                custom_prompt_tb,
                lang_dd,
                model_dd,
                tenant_selector,
                current_uid_state,
                authed,
                auth_session,
            ],
            outputs=[analysis_md],
            show_progress="full",
        )

    return demo

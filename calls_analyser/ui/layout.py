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
    with gr.Blocks(title="Vochi CRM Call Logs (Gradio)") as demo:
        gr.Markdown(
            "# Vochi CRM → MP3 → AI analysis\n"
            "*Filter calls by date, time and type, listen to recordings and run batch AI analysis.*"
        )

        authed = gr.State(False)
        batch_results_state = gr.State(pd.DataFrame())
        current_uid_state = gr.State("")
        custom_batch_conditions_state = gr.State(deps.batch_custom_conditions)

        with gr.Group(visible=os.environ.get("VOCHI_UI_PASSWORD", "") != "") as pwd_group:
            gr.Markdown("### 🔐 Enter password")
            pwd_tb = gr.Textbox(
                label="Password", type="password", placeholder="••••••••", lines=1
            )
            pwd_btn = gr.Button("Unlock", variant="primary")

        with gr.Tabs() as tabs:  # noqa: F841
            with gr.Tab("Vochi CRM"):
                with gr.Row():
                    tenant_tb = gr.Textbox(
                        label="Tenant ID", value="Amedis", scale=1
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
                        "Batch analyze custom", variant="secondary", scale=0
                    )
                    save_btn = gr.Button("Save to file", scale=0)

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

        pwd_btn.click(
            handlers.check_password,
            inputs=[pwd_tb],
            outputs=[authed, status_fetch, pwd_group],
        )

        filter_btn.click(
            handlers.filter_calls,
            inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, authed, tenant_tb],
            outputs=[calls_df, batch_results_df, row_dd, status_fetch, pwd_group],
        )

        batch_btn.click(
            fn=handlers.mass_analyze,
            inputs=[date_inp, time_from_inp, time_to_inp, call_type_dd, tenant_tb, authed],
            outputs=[batch_results_df, batch_status_md, batch_file],
        ).then(
            fn=lambda df: df,
            inputs=[batch_results_df],
            outputs=[batch_results_state],
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
            fn=handlers.mass_analyze_custom,
            inputs=[
                custom_batch_conditions_state,
                date_inp,
                time_from_inp,
                time_to_inp,
                call_type_dd,
                tenant_tb,
                authed,
            ],
            outputs=[batch_results_df, batch_status_md, batch_file],
        ).then(
            fn=lambda df: df,
            inputs=[batch_results_df],
            outputs=[batch_results_state],
        ).then(
            fn=handlers.hide_call_list,
            outputs=[calls_df],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        batch_results_df.select(
            fn=handlers.on_batch_row_select,
            inputs=[batch_results_df, batch_results_state, tenant_tb],
            outputs=[row_dd, current_uid_state, current_uid_md, url_html, audio_out, status_fetch],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        play_btn.click(
            handlers.play_audio,
            inputs=[row_dd, calls_df, tenant_tb],
            outputs=[url_html, audio_out, status_fetch],
        ).then(
            fn=None, js=JS_FIX_LINKS
        )

        save_btn.click(
            handlers.export_results,
            inputs=[batch_results_state],
            outputs=[batch_file, batch_status_md],
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
                tpl_dd,
                custom_prompt_tb,
                lang_dd,
                model_dd,
                tenant_tb,
                current_uid_state,
            ],
            outputs=[analysis_md],
            show_progress="full",
        )

    return demo

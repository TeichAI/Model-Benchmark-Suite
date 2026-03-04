import streamlit as st
import subprocess
import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
import io
import zipfile
import plotly.io as pio
import tempfile
import time
import signal
import datetime
from collections import deque

PLOTLY_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 800,
        "width": 1400,
        "scale": 2,
    }
}

PLOTLY_MMLU_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "height": 1200,
        "width": 2400,
        "scale": 3,
    }
}

st.set_page_config(page_title="TeichAI Benchmark Suite", layout="wide")

st.title("TeichAI Model Benchmark Suite")

results_placeholder = st.empty()

# --- Sidebar Configuration ---
st.sidebar.header("Configuration")

# Model Selection
default_model = "TeichAI/Qwen3-4B-Thinking-2507-Gemini-2.5-Flash-Distill"
models_input = st.sidebar.text_area(
    "Models (one per line)", value=default_model, height=100
)
models = [m.strip() for m in models_input.split("\n") if m.strip()]

# Benchmark (lm_eval task) Selection
benchmarks = st.sidebar.multiselect(
    "Benchmarks",
    [
        "gpqa_diamond_zeroshot",
        "gsm8k",
        "winogrande",
        "arc_challenge",
        "hellaswag",
        "truthfulqa_mc2",
        "mmlu",
        "ifeval",
        "humaneval",
    ],
    default=["gpqa_diamond_zeroshot"],
)

# DeepEval
run_deepeval = st.sidebar.checkbox("Run DeepEval (Qualitative Metrics)", value=False)
if run_deepeval and not os.getenv("OPENROUTER_API_KEY"):
    st.sidebar.warning("OPENROUTER_API_KEY not set. DeepEval may fail.")

# HuggingFace Token (for private models)
with st.sidebar.expander("HuggingFace Token"):
    hf_token = st.text_input(
        "HF Token",
        type="password",
        help="Enter your HuggingFace token to access private/gated models. "
        "Get one at https://huggingface.co/settings/tokens",
    )
    if hf_token:
        st.success("Token provided ✓")

# Settings
backend = st.sidebar.selectbox(
    "Inference Backend",
    ["hf", "vllm"],
    index=0,
    help="'hf' = HuggingFace Transformers (works everywhere). "
    "'vllm' = vLLM (Linux/WSL only, much faster for generation tasks like ifeval/humaneval).",
)
quantization = st.sidebar.selectbox("Quantization", ["4bit", "8bit", "none"], index=0)
vllm_max_model_len_default = int(os.getenv("VLLM_MAX_MODEL_LEN", "8192"))
vllm_max_model_len = st.sidebar.number_input(
    "vLLM Max Context Length",
    min_value=512,
    max_value=262144,
    value=vllm_max_model_len_default,
    step=512,
    disabled=backend != "vllm",
    help="Sets vLLM max_model_len to limit KV cache memory."
    " Only applies to vLLM backend.",
)
allow_code_eval = st.sidebar.checkbox(
    "Allow code execution (Humaneval/code_eval)",
    value=False,
    disabled="humaneval" not in benchmarks,
    help=(
        "Required for Humaneval/code_eval. Runs untrusted model code; "
        "enable only in a sandboxed environment."
    ),
)
apply_chat_template = st.sidebar.checkbox(
    "Apply chat template",
    value=True,
    help="Recommended for instruct/chat models to format prompts correctly.",
)
overwrite_saved = st.sidebar.checkbox("Overwrite saved results", value=False)

fewshot_mode = st.sidebar.selectbox(
    "Few-shot",
    ["Task default (recommended)", "Zero-shot (0)", "Custom"],
    index=0,
)
if fewshot_mode == "Zero-shot (0)":
    num_fewshot = 0
elif fewshot_mode == "Custom":
    num_fewshot = st.sidebar.number_input("num_fewshot", min_value=0, value=5)
else:
    num_fewshot = None

override_gen_kwargs = st.sidebar.checkbox(
    "Override generation settings (advanced)",
    value=False,
)

# Sampling Parameters
with st.sidebar.expander("Sampling Parameters"):
    do_sample = st.checkbox(
        "Enable sampling (not recommended for reproducible benchmarks)",
        value=False,
        disabled=not override_gen_kwargs,
    )
    temperature = st.slider(
        "Temperature",
        0.0,
        2.0,
        0.0,
        disabled=not override_gen_kwargs,
    )
    top_p = st.slider(
        "Top P",
        0.0,
        1.0,
        1.0,
        disabled=not override_gen_kwargs,
    )
    top_k = st.number_input(
        "Top K",
        value=0,
        min_value=0,
        disabled=not override_gen_kwargs,
    )
    repetition_penalty = st.slider(
        "Repetition Penalty",
        1.0,
        2.0,
        1.0,
        disabled=not override_gen_kwargs,
    )
    batch_size = st.number_input("Batch size (lm_eval)", min_value=1, value=1)

# Run / View Controls
view_saved_only = st.sidebar.checkbox(
    "View saved results only (no new runs)", value=False
)
run_clicked = st.sidebar.button("Run Benchmarks", type="primary")


def render_results(results_data):
    if not results_data:
        return

    st.divider()
    st.header("Results Comparison")

    df = pd.DataFrame(results_data)

    all_models = sorted(df["Model"].unique())
    all_benchmarks = sorted(df["Benchmark"].unique())

    selected_models = st.multiselect(
        "Models to display",
        options=all_models,
        default=all_models,
        key="results_filter_models",
    )
    selected_benchmarks = st.multiselect(
        "Benchmarks to display",
        options=all_benchmarks,
        default=all_benchmarks,
        key="results_filter_benchmarks",
    )

    st.session_state.selected_models_for_mmlu = selected_models

    filtered_df = df[
        df["Model"].isin(selected_models) & df["Benchmark"].isin(selected_benchmarks)
    ]

    if filtered_df.empty:
        st.info("No data for current selection.")
        return

    score_matrix = (
        filtered_df.pivot_table(
            index="Benchmark", columns="Model", values="Score", aggfunc="mean"
        )
        .reindex(columns=selected_models)
        .sort_index()
    )

    def _highlight_row_winners(row):
        valid = row.dropna()
        if valid.empty:
            return ["" for _ in row]
        winner = valid.max()
        return [
            "background-color: #dcfce7; font-weight: 700;"
            if pd.notna(v) and v == winner
            else ""
            for v in row
        ]

    def _score_to_text(v):
        return "-" if pd.isna(v) else f"{float(v):.3f}"

    def _score_to_md(v, is_winner):
        if pd.isna(v):
            return "-"
        value = f"{float(v):.3f}"
        return f"**{value}**" if is_winner else value

    def _safe_to_markdown(table_df):
        try:
            return table_df.to_markdown(index=False)
        except ImportError:
            return table_df.to_csv(index=False)

    matrix_md_rows = []
    for benchmark_name, row in score_matrix.iterrows():
        valid = row.dropna()
        winner_score = valid.max() if not valid.empty else None
        row_data = {"Benchmark": benchmark_name}
        for model_name in score_matrix.columns:
            value = row[model_name]
            is_winner = pd.notna(value) and winner_score is not None and value == winner_score
            row_data[model_name] = _score_to_md(value, is_winner)
        matrix_md_rows.append(row_data)

    score_matrix_md_df = pd.DataFrame(matrix_md_rows)

    st.subheader("Head-to-Head Score Matrix")
    st.caption(
        "Rows are benchmarks, columns are models. Best score in each row is highlighted."
    )
    st.dataframe(
        score_matrix.style.format("{:.3f}", na_rep="-").apply(
            _highlight_row_winners, axis=1
        ),
        use_container_width=True,
    )

    with st.expander("View matrix in markdown format"):
        st.markdown(_safe_to_markdown(score_matrix_md_df))

    base_model = st.selectbox(
        "Base model for win/loss analysis",
        options=selected_models,
        index=0,
        key="base_model_select",
    )
    compare_options = [m for m in selected_models if m != base_model]
    compare_models = st.multiselect(
        "Compare these models against base",
        options=compare_options,
        default=compare_options,
        key="compare_models_select",
    )

    comparison_summary_df = pd.DataFrame()
    benchmark_outcome_df = pd.DataFrame()
    key_takeaways = []

    if compare_models and base_model in score_matrix.columns:
        summary_rows = []
        for model_name in compare_models:
            if model_name not in score_matrix.columns:
                continue
            pair = score_matrix[[base_model, model_name]].dropna()
            if pair.empty:
                continue
            base_wins = int((pair[base_model] > pair[model_name]).sum())
            base_losses = int((pair[base_model] < pair[model_name]).sum())
            ties = int((pair[base_model] == pair[model_name]).sum())
            avg_delta = float((pair[base_model] - pair[model_name]).mean())
            summary_rows.append(
                {
                    "Model": model_name,
                    "Benchmarks Compared": int(len(pair)),
                    "Base Wins": base_wins,
                    "Base Losses": base_losses,
                    "Ties": ties,
                    "Avg Delta (Base-Model)": avg_delta,
                }
            )

        if summary_rows:
            comparison_summary_df = pd.DataFrame(summary_rows).sort_values(
                "Avg Delta (Base-Model)", ascending=False
            )

        outcome_rows = []
        for benchmark_name, row in score_matrix.iterrows():
            base_score = row.get(base_model)
            if pd.isna(base_score):
                continue
            competitor_scores = row[compare_models].dropna()
            if competitor_scores.empty:
                continue
            best_competitor = competitor_scores.idxmax()
            best_competitor_score = float(competitor_scores.max())
            delta = float(base_score - best_competitor_score)
            if delta > 0:
                outcome = "Win"
            elif delta < 0:
                outcome = "Loss"
            else:
                outcome = "Tie"
            outcome_rows.append(
                {
                    "Benchmark": benchmark_name,
                    "Base Score": float(base_score),
                    "Best Opponent": best_competitor,
                    "Opponent Score": best_competitor_score,
                    "Delta (Base-Opponent)": delta,
                    "Outcome": outcome,
                }
            )

        if outcome_rows:
            benchmark_outcome_df = pd.DataFrame(outcome_rows).sort_values("Benchmark")

        if not benchmark_outcome_df.empty:
            wins = int((benchmark_outcome_df["Outcome"] == "Win").sum())
            losses = int((benchmark_outcome_df["Outcome"] == "Loss").sum())
            ties = int((benchmark_outcome_df["Outcome"] == "Tie").sum())
            key_takeaways.append(
                f"{base_model}: {wins} wins, {losses} losses, {ties} ties across selected benchmarks."
            )
        if not comparison_summary_df.empty:
            strongest = comparison_summary_df.iloc[-1]
            weakest = comparison_summary_df.iloc[0]
            key_takeaways.append(
                f"Hardest competitor vs base: {strongest['Model']} (Avg Δ={strongest['Avg Delta (Base-Model)']:.3f})."
            )
            key_takeaways.append(
                f"Easiest competitor vs base: {weakest['Model']} (Avg Δ={weakest['Avg Delta (Base-Model)']:.3f})."
            )

    if not benchmark_outcome_df.empty:
        st.subheader("Where Base Model Wins/Loses")
        st.dataframe(
            benchmark_outcome_df.style.format(
                {
                    "Base Score": "{:.3f}",
                    "Opponent Score": "{:.3f}",
                    "Delta (Base-Opponent)": "{:.3f}",
                }
            ),
            use_container_width=True,
        )

    if not comparison_summary_df.empty:
        st.subheader("Base vs Variant Summary")
        st.dataframe(
            comparison_summary_df.style.format({"Avg Delta (Base-Model)": "{:.3f}"}),
            use_container_width=True,
        )

    if key_takeaways:
        st.subheader("Quick Read")
        for line in key_takeaways:
            st.markdown(f"- {line}")

    # Bar Chart
    fig = px.bar(
        filtered_df,
        x="Model",
        y="Score",
        color="Benchmark",
        barmode="group",
        title=f"Benchmark Results (Quant: {quantization})",
        text_auto=".2f",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1], fixedrange=True),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=120, l=60, r=60),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="results_bar_chart",
        config=PLOTLY_DOWNLOAD_CONFIG,
    )

    with st.expander("View long-form rows"):
        st.dataframe(
            filtered_df[
                ["Model", "Benchmark", "Score", "Total Questions", "Total Correct"]
            ],
            use_container_width=True,
        )

    # Raw Data Expander (full, unfiltered data)
    with st.expander("View Raw Results"):
        st.json(results_data)

    # Export to Markdown / ZIP / PDF (filtered view)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    display_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    md_content = "# Benchmark Results Report\n\n"
    md_content += f"**Date:** {display_timestamp}\n\n"
    md_content += "## Configuration\n"
    md_content += f"- **Quantization:** {quantization}\n"
    md_content += f"- **Temperature:** {temperature}\n"
    md_content += f"- **Top P:** {top_p}\n"
    md_content += f"- **Top K:** {top_k}\n"
    md_content += f"- **Repetition Penalty:** {repetition_penalty}\n"
    md_content += f"- **Base Model:** {base_model}\n\n"

    md_content += "## Head-to-Head Score Matrix\n\n"
    md_content += _safe_to_markdown(score_matrix_md_df)

    if not benchmark_outcome_df.empty:
        md_content += "\n\n## Where Base Model Wins/Loses\n\n"
        benchmark_outcome_export = benchmark_outcome_df.copy()
        md_content += _safe_to_markdown(benchmark_outcome_export)

    if not comparison_summary_df.empty:
        md_content += "\n\n## Base vs Variant Summary\n\n"
        md_content += _safe_to_markdown(comparison_summary_df)

    if key_takeaways:
        md_content += "\n\n## Quick Read\n\n"
        for line in key_takeaways:
            md_content += f"- {line}\n"

    md_content += "\n\n## Full Row Data\n\n"
    md_content += _safe_to_markdown(
        filtered_df[["Model", "Benchmark", "Score", "Total Questions", "Total Correct"]]
    )

    mmlu_filtered = None
    mmlu_subject_results = st.session_state.get("mmlu_subject_results")
    if mmlu_subject_results:
        mmlu_df = pd.DataFrame(mmlu_subject_results)
        mmlu_df["Subject"] = mmlu_df["Benchmark"].apply(
            lambda b: (
                b[len("mmlu_") :] if isinstance(b, str) and b.startswith("mmlu_") else b
            )
        )
        mmlu_filtered = mmlu_df[mmlu_df["Model"].isin(selected_models)]
        if not mmlu_filtered.empty:
            md_content += "\n\n## MMLU Subject Breakdown\n\n"
            md_content += (
                '![alt="MMLU Subject Breakdown"](mmlu_subject_breakdown.png)\n\n'
            )
            mmlu_export = mmlu_filtered[
                [
                    "Model",
                    "Subject",
                    "Benchmark",
                    "Score",
                    "Total Questions",
                    "Total Correct",
                ]
            ].copy()
            try:
                md_mmlu_table = mmlu_export.to_markdown(index=False)
            except ImportError:
                md_mmlu_table = mmlu_export.to_csv(index=False)
            md_content += md_mmlu_table

    # Generate chart images for ZIP/PDF exports
    results_image_bytes = None
    try:
        # Use a light template and explicit background for exported charts
        fig_export = go.Figure(fig)
        fig_export.update_layout(
            template="plotly_white",
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(color="black"),
        )
        results_image_bytes = pio.to_image(
            fig_export,
            format="png",
            width=PLOTLY_DOWNLOAD_CONFIG["toImageButtonOptions"]["width"],
            height=PLOTLY_DOWNLOAD_CONFIG["toImageButtonOptions"]["height"],
            scale=PLOTLY_DOWNLOAD_CONFIG["toImageButtonOptions"]["scale"],
        )
    except Exception as e:
        results_image_bytes = None
        st.warning(
            f"Failed to generate main results chart image for PDF/ZIP export: {e}"
        )

    mmlu_image_bytes = None
    if mmlu_filtered is not None and not mmlu_filtered.empty:
        try:
            fig_mmlu_export = px.bar(
                mmlu_filtered,
                x="Subject",
                y="Score",
                color="Model",
                barmode="group",
                title="MMLU Subject Scores",
                text_auto=".2f",
            )
            fig_mmlu_export.update_layout(
                template="plotly_white",
                paper_bgcolor="white",
                plot_bgcolor="white",
                font=dict(color="black"),
                yaxis=dict(range=[0, 1], fixedrange=True),
                xaxis_tickangle=-45,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(t=80, b=150, l=60, r=60),
            )
            mmlu_image_bytes = pio.to_image(
                fig_mmlu_export,
                format="png",
                width=PLOTLY_MMLU_DOWNLOAD_CONFIG["toImageButtonOptions"]["width"],
                height=PLOTLY_MMLU_DOWNLOAD_CONFIG["toImageButtonOptions"]["height"],
                scale=PLOTLY_MMLU_DOWNLOAD_CONFIG["toImageButtonOptions"]["scale"],
            )
        except Exception as e:
            mmlu_image_bytes = None
            st.warning(f"Failed to generate MMLU chart image for PDF/ZIP export: {e}")

    zip_bytes = None
    zip_error = None
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", md_content)
            if results_image_bytes is not None:
                zf.writestr("results_bar_chart.png", results_image_bytes)
            if mmlu_image_bytes is not None:
                zf.writestr("mmlu_subject_breakdown.png", mmlu_image_bytes)
        zip_bytes = zip_buffer.getvalue()
    except Exception as e:
        zip_bytes = None
        zip_error = str(e)

    pdf_bytes = None
    pdf_error = None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Image as RLImage,
            PageBreak,
            Table,
            TableStyle,
        )
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors

        pdf_buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            pdf_buffer,
            pagesize=letter,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50,
        )

        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            "TeichTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=colors.HexColor("#2563EB"),
            alignment=1,
            spaceAfter=16,
        )
        heading_style = ParagraphStyle(
            "SectionHeading",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=14,
            spaceBefore=12,
            spaceAfter=6,
        )
        body_style = ParagraphStyle(
            "Body",
            parent=styles["Normal"],
            fontName="Helvetica",
            fontSize=10,
            leading=13,
        )

        def _build_table(table_df, highlight_winners=False):
            cols = list(table_df.columns)
            data = [cols]
            for _, row in table_df.iterrows():
                data.append([str(row[c]) for c in cols])

            report_table = Table(data, repeatRows=1)
            style_cmds = [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F2937")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#D1D5DB")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F9FAFB")]),
            ]

            if highlight_winners and cols and cols[0] == "Benchmark":
                model_cols = cols[1:]
                for row_idx, (_, row) in enumerate(table_df.iterrows(), start=1):
                    numeric_scores = []
                    for model_name in model_cols:
                        try:
                            numeric_scores.append(float(row[model_name]))
                        except Exception:
                            numeric_scores.append(float("nan"))
                    clean_scores = [s for s in numeric_scores if pd.notna(s)]
                    if not clean_scores:
                        continue
                    winner_score = max(clean_scores)
                    for col_idx, score in enumerate(numeric_scores, start=1):
                        if pd.notna(score) and score == winner_score:
                            style_cmds.append(
                                (
                                    "FONTNAME",
                                    (col_idx, row_idx),
                                    (col_idx, row_idx),
                                    "Helvetica-Bold",
                                )
                            )
                            style_cmds.append(
                                (
                                    "BACKGROUND",
                                    (col_idx, row_idx),
                                    (col_idx, row_idx),
                                    colors.HexColor("#DCFCE7"),
                                )
                            )

            report_table.setStyle(TableStyle(style_cmds))
            return report_table

        elements = []

        # Title
        elements.append(Paragraph("TeichAI Benchmark Results Report", title_style))

        # Configuration block
        elements.append(Paragraph(f"<b>Date:</b> {display_timestamp}", body_style))
        elements.append(Paragraph(f"<b>Quantization:</b> {quantization}", body_style))
        elements.append(Paragraph(f"<b>Temperature:</b> {temperature}", body_style))
        elements.append(Paragraph(f"<b>Top P:</b> {top_p}", body_style))
        elements.append(Paragraph(f"<b>Top K:</b> {top_k}", body_style))
        elements.append(
            Paragraph(f"<b>Repetition Penalty:</b> {repetition_penalty}", body_style)
        )
        elements.append(Paragraph(f"<b>Base Model:</b> {base_model}", body_style))

        if key_takeaways:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Quick Read", heading_style))
            for line in key_takeaways:
                elements.append(Paragraph(f"• {line}", body_style))

        elements.append(Spacer(1, 12))
        elements.append(Paragraph("Head-to-Head Score Matrix", heading_style))
        matrix_pdf = score_matrix.reset_index().copy()
        for model_name in selected_models:
            if model_name in matrix_pdf.columns:
                matrix_pdf[model_name] = matrix_pdf[model_name].map(_score_to_text)
        elements.append(_build_table(matrix_pdf, highlight_winners=True))

        if not benchmark_outcome_df.empty:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Where Base Model Wins/Loses", heading_style))
            outcome_pdf = benchmark_outcome_df.copy()
            for col in ["Base Score", "Opponent Score", "Delta (Base-Opponent)"]:
                outcome_pdf[col] = outcome_pdf[col].map(lambda v: f"{float(v):.3f}")
            elements.append(_build_table(outcome_pdf))

        if not comparison_summary_df.empty:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Base vs Variant Summary", heading_style))
            summary_pdf = comparison_summary_df.copy()
            summary_pdf["Avg Delta (Base-Model)"] = summary_pdf[
                "Avg Delta (Base-Model)"
            ].map(lambda v: f"{float(v):.3f}")
            elements.append(_build_table(summary_pdf))

        if results_image_bytes is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("Benchmark Results", heading_style))
            elements.append(Spacer(1, 6))
            img_buffer = io.BytesIO(results_image_bytes)
            img = RLImage(img_buffer)
            img._restrictSize(doc.width, doc.height - 100)
            img.hAlign = "CENTER"
            elements.append(img)

        if mmlu_image_bytes is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("MMLU Subject Breakdown", heading_style))
            elements.append(Spacer(1, 6))
            img_buffer = io.BytesIO(mmlu_image_bytes)
            img = RLImage(img_buffer)
            img._restrictSize(doc.width, doc.height - 100)
            img.hAlign = "CENTER"
            elements.append(img)

        doc.build(elements)
        pdf_bytes = pdf_buffer.getvalue()
    except Exception as e:
        pdf_bytes = None
        pdf_error = str(e)

    st.download_button(
        label="Download Clean Markdown Report",
        data=md_content,
        file_name=f"benchmark_report_{timestamp}.md",
        mime="text/markdown",
    )

    if zip_bytes is not None:
        st.download_button(
            label="Download Markdown ZIP (report + charts)",
            data=zip_bytes,
            file_name=f"benchmark_report_{timestamp}.zip",
            mime="application/zip",
        )
    else:
        st.info(
            "ZIP export failed. "
            f"Reason: {zip_error or 'unknown error'}."
        )

    if pdf_bytes is not None:
        st.download_button(
            label="Download PDF Report",
            data=pdf_bytes,
            file_name=f"benchmark_report_{timestamp}.pdf",
            mime="application/pdf",
        )
    else:
        st.info(
            "PDF export failed. "
            f"Reason: {pdf_error or 'unknown error'}. "
            "Install/update `reportlab` (and `kaleido` if you want chart images)."
        )

    # --- DeepEval Qualitative Analysis ---
    if run_deepeval:
        st.divider()
        st.header("DeepEval Qualitative Analysis")

        deepeval_data = []
        for item in results_data:
            model = item["Model"]
            benchmark = item["Benchmark"]
            safe_model = model.replace("/", "_")
            # Preferred: new location under saved_results, matching lm_eval raw outputs
            result_file = os.path.join(
                "saved_results",
                f"results_raw_{safe_model}_{benchmark}_deepeval.json",
            )

            # Backwards compatibility: fall back to old root-level naming if needed
            if not os.path.exists(result_file):
                legacy_file = f"results_{safe_model}_{benchmark}_deepeval.json"
                if os.path.exists(legacy_file):
                    result_file = legacy_file
                else:
                    continue

            if os.path.exists(result_file):
                with open(result_file, "r") as f:
                    try:
                        eval_results = json.load(f)
                        for res in eval_results:
                            deepeval_data.append(
                                {
                                    "Model": model,
                                    "Benchmark": benchmark,
                                    "Input": res.get("input", ""),
                                    "Score": res.get("score", 0),
                                    "Reason": res.get("reason", ""),
                                }
                            )
                    except json.JSONDecodeError:
                        st.warning(
                            f"Could not parse DeepEval results for {model} on {benchmark}"
                        )

        if deepeval_data:
            df_deep = pd.DataFrame(deepeval_data)

            # Average Score Chart
            avg_scores = (
                df_deep.groupby(["Model", "Benchmark"])["Score"].mean().reset_index()
            )
            fig_deep = px.bar(
                avg_scores,
                x="Model",
                y="Score",
                color="Benchmark",
                title="Average DeepEval Relevancy Score",
                text_auto=".2f",
            )
            fig_deep.update_layout(
                yaxis=dict(range=[0, 1], fixedrange=True),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="center",
                    x=0.5,
                ),
                margin=dict(t=80, b=120, l=60, r=60),
            )
            st.plotly_chart(
                fig_deep,
                use_container_width=True,
                key="deepeval_bar_chart",
                config=PLOTLY_DOWNLOAD_CONFIG,
            )

            # Detailed Table
            st.subheader("Detailed Qualitative Feedback")
            st.dataframe(df_deep)


def render_mmlu_breakdown(mmlu_results):
    if not mmlu_results:
        return

    st.divider()
    st.header("MMLU Subject Breakdown")

    df = pd.DataFrame(mmlu_results)
    df["Subject"] = df["Benchmark"].apply(
        lambda b: (
            b[len("mmlu_") :] if isinstance(b, str) and b.startswith("mmlu_") else b
        )
    )

    global_selected_models = st.session_state.get("selected_models_for_mmlu")
    if isinstance(global_selected_models, list) and global_selected_models:
        df = df[df["Model"].isin(global_selected_models)]

    all_models = sorted(df["Model"].unique())
    all_subjects = sorted(df["Subject"].unique())

    selected_models = st.multiselect(
        "Models to display (MMLU)",
        options=all_models,
        default=all_models,
        key="mmlu_filter_models",
    )
    selected_subjects = st.multiselect(
        "MMLU subjects to display",
        options=all_subjects,
        default=all_subjects,
        key="mmlu_filter_subjects",
    )

    filtered_df = df[
        df["Model"].isin(selected_models) & df["Subject"].isin(selected_subjects)
    ]

    if filtered_df.empty:
        st.info("No MMLU subject data for current selection.")
        return

    fig = px.bar(
        filtered_df,
        x="Subject",
        y="Score",
        color="Model",
        barmode="group",
        title="MMLU Subject Scores",
        text_auto=".2f",
    )
    fig.update_layout(
        yaxis=dict(range=[0, 1], fixedrange=True),
        xaxis_tickangle=-45,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=80, b=150, l=60, r=60),
    )
    st.plotly_chart(
        fig,
        use_container_width=True,
        key="mmlu_subject_bar_chart",
        config=PLOTLY_MMLU_DOWNLOAD_CONFIG,
    )

    st.dataframe(
        filtered_df[
            [
                "Model",
                "Subject",
                "Benchmark",
                "Score",
                "Total Questions",
                "Total Correct",
            ]
        ]
    )


def summarize_results(model, benchmark, data):
    # Extract score and details
    score = 0
    total_questions = 0
    total_correct = 0

    # lm_eval tasks: benchmark is the lm_eval task name
    lm_data = data.get("lm_eval", {})
    if not isinstance(lm_data, dict):
        return score, total_questions, total_correct

    lm_res = lm_data.get("results", {})
    task_metrics = lm_res.get(benchmark, {})

    if isinstance(task_metrics, dict):
        # Try common metric keys in priority order across different benchmark types:
        # - acc,none / acc_norm,none: standard multiple-choice benchmarks
        # - exact_match,none: gsm8k and similar
        # - prompt_level_strict_acc,none: ifeval
        # - pass@1,none: humaneval (code generation)
        metric_keys = [
            "acc,none",
            "acc_norm,none",
            "exact_match,none",
            "prompt_level_strict_acc,none",
            "pass@1,create_test",
            "pass@1,none",
            "pass@1",
        ]
        task_score = None
        for key in metric_keys:
            task_score = task_metrics.get(key)
            if task_score is not None:
                break
        if task_score is None:
            task_score = 0
        score = task_score

    n_samples_dict = lm_data.get("n-samples", {})
    group_subtasks = lm_data.get("group_subtasks", {})

    def resolve_n_samples(task: str, visited: set | None = None) -> int:
        """Return total number of samples for a task.

        For simple tasks we read lm_eval["n-samples"][task]. For grouped
        tasks like MMLU aggregates (e.g. "mmlu", "mmlu_humanities"), we
        recursively sum over their subtasks from group_subtasks.
        """

        if visited is None:
            visited = set()
        if task in visited:
            return 0
        visited.add(task)

        count_data = n_samples_dict.get(task)
        if isinstance(count_data, dict):
            eff = count_data.get("effective", count_data.get("original", 0))
            if isinstance(eff, (int, float)):
                return int(eff)
        elif isinstance(count_data, (int, float)):
            return int(count_data)

        # Fall back to summing over child tasks if this is a group key
        children = group_subtasks.get(task)
        if isinstance(children, list):
            return sum(resolve_n_samples(child, visited) for child in children)

        return 0

    total_questions = resolve_n_samples(benchmark)
    total_correct = int(score * total_questions) if total_questions else 0

    return score, total_questions, total_correct


def get_run_config(model, benchmark):
    return {
        "model": model,
        "benchmark": benchmark,
        "backend": backend,
        "quantization": quantization,
        "max_model_len": int(vllm_max_model_len) if backend == "vllm" else None,
        "allow_code_eval": bool(allow_code_eval),
        "apply_chat_template": bool(apply_chat_template),
        "num_fewshot": None if num_fewshot is None else int(num_fewshot),
        "override_gen_kwargs": bool(override_gen_kwargs),
        "do_sample": bool(do_sample) if override_gen_kwargs else False,
        "temperature": float(temperature) if override_gen_kwargs else None,
        "top_p": float(top_p) if override_gen_kwargs else None,
        "top_k": int(top_k) if override_gen_kwargs else None,
        "repetition_penalty": float(repetition_penalty) if override_gen_kwargs else None,
        "batch_size": int(batch_size),
    }


def get_cache_path(model, benchmark):
    safe_model = model.replace("/", "_")
    return os.path.join("saved_results", f"results_{safe_model}_{benchmark}.json")


def load_all_saved_results():
    """Load all lm_eval results from raw JSON files in saved_results/ without running benchmarks."""
    summary_results = []
    mmlu_subject_results = []
    saved_dir = "saved_results"

    if not os.path.isdir(saved_dir):
        return summary_results, mmlu_subject_results

    for fname in os.listdir(saved_dir):
        if not (fname.startswith("results_raw_") and fname.endswith(".json")):
            continue

        path = os.path.join(saved_dir, fname)
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        lm_data = data.get("lm_eval", {})
        if not isinstance(lm_data, dict):
            continue

        lm_results = lm_data.get("results", {})
        if not isinstance(lm_results, dict) or not lm_results:
            continue

        for benchmark in lm_results.keys():
            model = None

            configs = lm_data.get("configs", {})
            if isinstance(configs, dict):
                task_cfg = configs.get(benchmark)
                if isinstance(task_cfg, dict):
                    metadata = task_cfg.get("metadata") or {}
                    pretrained = metadata.get("pretrained")
                    if isinstance(pretrained, str) and pretrained:
                        model = pretrained

            if not model:
                base = os.path.splitext(fname)[0]
                prefix = "results_raw_"
                if base.startswith(prefix):
                    core = base[len(prefix) :]
                    suffix = f"_{benchmark}"
                    if core.endswith(suffix):
                        safe_model = core[: -len(suffix)]
                        if safe_model:
                            model = safe_model.replace("_", "/")

            if not model:
                # Skip unresolved models instead of labeling them as "unknown_model".
                continue

            score, total_questions, total_correct = summarize_results(
                model, benchmark, data
            )

            entry = {
                "Model": model,
                "Benchmark": benchmark,
                "Score": score,
                "Total Questions": int(total_questions),
                "Total Correct": int(total_correct),
                "Details": data,
            }

            if isinstance(benchmark, str) and benchmark.startswith("mmlu_"):
                mmlu_subject_results.append(entry)
            else:
                summary_results.append(entry)

    return summary_results, mmlu_subject_results


# --- Main Execution ---

if view_saved_only:
    summary_results, mmlu_subject_results = load_all_saved_results()
    if not summary_results and not mmlu_subject_results:
        st.info("No saved results found in 'saved_results' directory.")
    else:
        st.session_state.results = summary_results
        st.session_state.mmlu_subject_results = mmlu_subject_results
        with results_placeholder.container():
            if summary_results:
                render_results(summary_results)
            if mmlu_subject_results:
                render_mmlu_breakdown(mmlu_subject_results)

elif run_clicked:
    if not models:
        st.error("Please specify at least one model.")
    elif not benchmarks:
        st.error("Please select at least one benchmark.")
    else:
        progress_bar = st.progress(0)
        status_text = st.empty()

        results_list = []

        total_steps = len(models) * len(benchmarks)
        current_step = 0

        os.makedirs("saved_results", exist_ok=True)

        for model in models:
            for benchmark in benchmarks:
                current_step += 1
                progress = current_step / total_steps
                progress_bar.progress(progress)

                cache_path = get_cache_path(model, benchmark)

                if os.path.exists(cache_path) and not overwrite_saved:
                    try:
                        with open(cache_path, "r") as f:
                            cached_payload = json.load(f)

                        if (
                            isinstance(cached_payload, dict)
                            and "data" in cached_payload
                        ):
                            cached_config = cached_payload.get("config")
                            data = cached_payload.get("data")
                        else:
                            # Backwards compatibility: cache file is raw data
                            cached_config = None
                            data = cached_payload

                        current_config = get_run_config(model, benchmark)
                        # Intentionally ignore differences between cached_config and
                        # current_config so we always reuse cached results.

                        status_text.text(
                            f"Using cached results for {benchmark.upper()} on {model}"
                        )

                        score, total_questions, total_correct = summarize_results(
                            model, benchmark, data
                        )
                        results_list.append(
                            {
                                "Model": model,
                                "Benchmark": benchmark,
                                "Score": score,
                                "Total Questions": int(total_questions),
                                "Total Correct": int(total_correct),
                                "Details": data,
                            }
                        )
                        st.session_state.results = results_list
                        continue
                    except Exception as e:
                        st.warning(
                            f"Failed to use cached results for {model} on {benchmark}: {e}. Rerunning."
                        )

                status_text.text(f"Running {benchmark.upper()} on {model}...")

                # Determine path to main.py
                script_path = "main.py" if os.path.exists("main.py") else "eval/main.py"
                if not os.path.exists(script_path):
                    st.error(f"Could not find main.py at {script_path}")
                    continue

                # Construct command: always use lm_eval as framework and this benchmark as the task
                safe_model = model.replace("/", "_")
                output_file = os.path.join(
                    "saved_results", f"results_raw_{safe_model}_{benchmark}.json"
                )
                cmd = [
                    sys.executable,
                    script_path,
                    "--model",
                    model,
                    "--benchmark",
                    "lm_eval",
                    "--tasks",
                    benchmark,
                    "--backend",
                    backend,
                    "--quantization",
                    quantization,
                ]

                if backend == "vllm":
                    cmd.extend(["--max_model_len", str(int(vllm_max_model_len))])

                cmd += [
                    "--batch_size",
                    str(int(batch_size)),
                    "--output",
                    output_file,
                ]

                if num_fewshot is not None:
                    cmd.extend(["--num_fewshot", str(int(num_fewshot))])

                if override_gen_kwargs:
                    cmd.append("--override_gen_kwargs")
                    if do_sample:
                        cmd.append("--do_sample")
                    cmd += [
                        "--temperature",
                        str(temperature),
                        "--top_p",
                        str(top_p),
                        "--top_k",
                        str(top_k),
                        "--repetition_penalty",
                        str(repetition_penalty),
                    ]

                if run_deepeval:
                    cmd.append("--deepeval")

                if hf_token:
                    cmd.extend(["--hf_token", hf_token])

                if allow_code_eval:
                    cmd.append("--allow_code_eval")

                if apply_chat_template:
                    cmd.append("--apply_chat_template")

                # Run subprocess with real-time logging
                process = None
                log_path = None
                try:
                    with tempfile.NamedTemporaryFile(
                        mode="w+", suffix=".log", delete=False
                    ) as log_file:
                        log_path = log_file.name

                        process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            universal_newlines=True,
                            start_new_session=True,
                        )

                        recent_lines = deque(maxlen=200)
                        latest_line = ""
                        last_ui_update = 0.0
                        line_count = 0

                        with st.spinner(
                            f"Running **{benchmark}** for **{model}**:"):
                            log_placeholder = st.empty()

                            while True:
                                line = process.stdout.readline()
                                if not line and process.poll() is not None:
                                    break
                                if line:
                                    log_file.write(line)
                                    line_count += 1
                                    if line_count % 20 == 0:
                                        log_file.flush()
                                    candidate = line.rstrip()
                                    if candidate:
                                        recent_lines.append(candidate)
                                        latest_line = candidate
                                        now = time.time()
                                        if now - last_ui_update > 0.25:
                                            log_placeholder.code(
                                                latest_line, language="bash"
                                            )
                                            last_ui_update = now

                        # Clear live log line once the run is finished
                        log_placeholder.empty()

                        # Ensure log is flushed
                        log_file.flush()

                    # Join all logs for download (read from temp file)
                    final_logs = ""
                    if log_path and os.path.exists(log_path):
                        with open(log_path, "r", encoding="utf-8") as f:
                            final_logs = f.read()

                    # Clean up temp log file
                    if log_path and os.path.exists(log_path):
                        try:
                            os.remove(log_path)
                        except OSError:
                            pass

                    if process.returncode != 0:
                        st.error(f"Error running {benchmark} on {model}")
                        if recent_lines:
                            st.code("\n".join(recent_lines), language="bash")
                        st.download_button(
                            label="Download Full Logs (Error)",
                            data=final_logs,
                            file_name=f"logs_{model.replace('/', '_')}_{benchmark}_error.txt",
                            mime="text/plain",
                        )
                        continue

                    st.success(f"Finished **{benchmark}** for **{model}**:")
                    st.download_button(
                        label="Download Full Logs",
                        data=final_logs,
                        file_name=f"logs_{model.replace('/', '_')}_{benchmark}.txt",
                        mime="text/plain",
                    )

                    result_file = output_file
                    if not os.path.exists(result_file):
                        st.error(
                            f"Expected result file {result_file} not found for {model} on {benchmark}"
                        )
                        continue

                    with open(result_file, "r") as f:
                        data = json.load(f)

                    # Save a copy into cache with configuration
                    try:
                        cache_payload = {
                            "config": get_run_config(model, benchmark),
                            "data": data,
                        }
                        with open(cache_path, "w") as f:
                            json.dump(cache_payload, f)
                    except Exception as e:
                        st.warning(
                            f"Failed to write cached results for {model} on {benchmark}: {e}"
                        )

                    score, total_questions, total_correct = summarize_results(
                        model, benchmark, data
                    )

                    results_list.append(
                        {
                            "Model": model,
                            "Benchmark": benchmark,
                            "Score": score,
                            "Total Questions": int(total_questions),
                            "Total Correct": int(total_correct),
                            "Details": data,
                        }
                    )

                    st.session_state.results = results_list

                except Exception as e:
                    st.error(f"Execution failed: {e}")
                    import traceback

                    st.code(traceback.format_exc())
                finally:
                    # Ensure subprocess is terminated and pipes are closed
                    try:
                        if process and process.poll() is None:
                            try:
                                os.killpg(process.pid, signal.SIGTERM)
                            except Exception:
                                process.terminate()
                            try:
                                process.wait(timeout=10)
                            except Exception:
                                try:
                                    os.killpg(process.pid, signal.SIGKILL)
                                except Exception:
                                    process.kill()
                        if process and process.stdout:
                            process.stdout.close()
                    except Exception:
                        pass

        status_text.text("All benchmarks completed!")
        st.session_state.results = results_list
        with results_placeholder.container():
            render_results(results_list)


# --- Visualization ---
elif "results" in st.session_state and st.session_state.results:
    with results_placeholder.container():
        render_results(st.session_state.results)
        if (
            "mmlu_subject_results" in st.session_state
            and st.session_state.mmlu_subject_results
        ):
            render_mmlu_breakdown(st.session_state.mmlu_subject_results)

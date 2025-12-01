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
    ],
    default=["gpqa_diamond_zeroshot"],
)

# DeepEval
run_deepeval = st.sidebar.checkbox("Run DeepEval (Qualitative Metrics)", value=False)
if run_deepeval and not os.getenv("OPENROUTER_API_KEY"):
    st.sidebar.warning("OPENROUTER_API_KEY not set. DeepEval may fail.")

# Settings
quantization = st.sidebar.selectbox("Quantization", ["4bit", "8bit", "none"], index=0)
overwrite_saved = st.sidebar.checkbox("Overwrite saved results", value=False)

# Sampling Parameters
with st.sidebar.expander("Sampling Parameters"):
    temperature = st.slider("Temperature", 0.0, 2.0, 0.6)
    top_p = st.slider("Top P", 0.0, 1.0, 0.95)
    top_k = st.number_input("Top K", value=20)
    repetition_penalty = st.slider("Repetition Penalty", 1.0, 2.0, 1.1)
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

    key_takeaways = []

    base_model = st.selectbox(
        "Base model for comparison",
        options=selected_models,
        index=0,
        key="base_model_select",
    )
    compare_options = [m for m in selected_models if m != base_model]
    comparison_df = None
    comparison_summary_df = None
    if compare_options:
        compare_models = st.multiselect(
            "Models to compare vs base",
            options=compare_options,
            default=compare_options,
            key="compare_models_select",
        )
        if compare_models:
            try:
                pivot = filtered_df.pivot_table(
                    index="Benchmark", columns="Model", values="Score", aggfunc="mean"
                )
                if base_model in pivot.columns:
                    rows = []
                    for model in compare_models:
                        if model not in pivot.columns:
                            continue
                        sub = pd.DataFrame(
                            {
                                "Benchmark": pivot.index,
                                "Base Model": base_model,
                                "Compare Model": model,
                                "Base Score": pivot[base_model],
                                "Model Score": pivot[model],
                            }
                        ).dropna()
                        if not sub.empty:
                            sub["Delta"] = sub["Model Score"] - sub["Base Score"]
                            denom = sub["Base Score"].where(sub["Base Score"] != 0)
                            sub["Delta %"] = sub["Delta"] / denom
                            rows.append(sub)
                    if rows:
                        comparison_df = pd.concat(rows, ignore_index=True)
                        summary_rows = []
                        for model, group in comparison_df.groupby("Compare Model"):
                            wins = (group["Delta"] > 0).sum()
                            ties = (group["Delta"] == 0).sum()
                            losses = (group["Delta"] < 0).sum()
                            avg_delta = group["Delta"].mean()
                            summary_rows.append(
                                {
                                    "Compare Model": model,
                                    "Benchmarks Compared": int(len(group)),
                                    "Wins vs Base": int(wins),
                                    "Ties vs Base": int(ties),
                                    "Losses vs Base": int(losses),
                                    "Avg Delta": float(avg_delta),
                                }
                            )
                        if summary_rows:
                            summary_df = pd.DataFrame(summary_rows)
                            summary_df = summary_df.sort_values(
                                "Avg Delta", ascending=False
                            )
                            comparison_summary_df = summary_df

                            key_takeaways = []

                            def _fmt_pct(v):
                                return f"{v:.1%}" if pd.notna(v) else "n/a"

                            if not summary_df.empty:
                                top = summary_df.iloc[0]
                                key_takeaways.append(
                                    f"Overall vs {base_model}, {top['Compare Model']} has the best average score "
                                    f"(Avg Δ={top['Avg Delta']:.3f} across {int(top['Benchmarks Compared'])} benchmarks)."
                                )
                                if len(summary_df) > 1:
                                    bottom = summary_df.iloc[-1]
                                    desc = (
                                        "worst average score"
                                        if bottom["Avg Delta"] < 0
                                        else "lowest average gain"
                                    )
                                    key_takeaways.append(
                                        f"Overall vs {base_model}, {bottom['Compare Model']} has the {desc} "
                                        f"(Avg Δ={bottom['Avg Delta']:.3f} across {int(bottom['Benchmarks Compared'])} benchmarks)."
                                    )

                            for model in summary_df["Compare Model"]:
                                group = comparison_df[
                                    comparison_df["Compare Model"] == model
                                ]
                                if group.empty:
                                    continue
                                best_idx = group["Delta"].idxmax()
                                worst_idx = group["Delta"].idxmin()
                                best = group.loc[best_idx]
                                worst = group.loc[worst_idx]
                                key_takeaways.append(
                                    f"For {model} vs {base_model}, the largest gain is on {best['Benchmark']} "
                                    f"(Δ={best['Delta']:.3f}, rel={_fmt_pct(best['Delta %'])}), while the largest drop is on {worst['Benchmark']} "
                                    f"(Δ={worst['Delta']:.3f}, rel={_fmt_pct(worst['Delta %'])})."
                                )
            except Exception:
                comparison_df = None
                comparison_summary_df = None
    if comparison_df is not None and not comparison_df.empty:
        st.subheader("Model Comparison vs Base")
        st.dataframe(
            comparison_df[
                [
                    "Compare Model",
                    "Benchmark",
                    "Base Score",
                    "Model Score",
                    "Delta",
                    "Delta %",
                ]
            ]
        )
    if comparison_summary_df is not None and not comparison_summary_df.empty:
        st.dataframe(comparison_summary_df)

    if key_takeaways:
        st.subheader("Key Takeaways")
        for line in key_takeaways[:6]:
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

    # Data Table
    st.dataframe(
        filtered_df[["Model", "Benchmark", "Score", "Total Questions", "Total Correct"]]
    )

    # Raw Data Expander (full, unfiltered data)
    with st.expander("View Raw Results"):
        st.json(results_data)

    # Export to Markdown / ZIP / PDF (filtered view)
    import datetime

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    display_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Generate Markdown Content
    md_content = f"# Benchmark Results Report\n\n"
    md_content += f"**Date:** {display_timestamp}\n\n"
    md_content += f"## Configuration\n"
    md_content += f"- **Quantization:** {quantization}\n"
    md_content += f"- **Temperature:** {temperature}\n"
    md_content += f"- **Top P:** {top_p}\n"
    md_content += f"- **Top K:** {top_k}\n"
    md_content += f"- **Repetition Penalty:** {repetition_penalty}\n\n"

    md_content += "## Results\n\n"
    md_content += '![alt="Results Bar Chart"](results_bar_chart.png)\n\n'

    md_content += "## Detailed Results\n\n"
    try:
        md_table = filtered_df[
            ["Model", "Benchmark", "Score", "Total Questions", "Total Correct"]
        ].to_markdown(index=False)
    except ImportError:
        md_table = filtered_df[
            ["Model", "Benchmark", "Score", "Total Questions", "Total Correct"]
        ].to_csv(index=False)
    md_content += md_table

    if comparison_df is not None and not comparison_df.empty:
        md_content += "\n\n## Model Comparison vs Base\n\n"
        md_content += f"- Base model: {base_model}\n\n"
        comp_export = comparison_df[
            [
                "Compare Model",
                "Benchmark",
                "Base Score",
                "Model Score",
                "Delta",
                "Delta %",
            ]
        ].copy()
        try:
            md_comp_table = comp_export.to_markdown(index=False)
        except ImportError:
            md_comp_table = comp_export.to_csv(index=False)
        md_content += md_comp_table

        if comparison_summary_df is not None and not comparison_summary_df.empty:
            md_content += "\n\n### Aggregate Comparison\n\n"
            try:
                md_summary_table = comparison_summary_df.to_markdown(index=False)
            except ImportError:
                md_summary_table = comparison_summary_df.to_csv(index=False)
            md_content += md_summary_table

    if key_takeaways:
        md_content += "\n\n## Key Takeaways\n\n"
        for line in key_takeaways[:6]:
            md_content += f"- {line}\n"

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
    try:
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("report.md", md_content)
            if results_image_bytes is not None:
                zf.writestr("results_bar_chart.png", results_image_bytes)
            if mmlu_image_bytes is not None:
                zf.writestr("mmlu_subject_breakdown.png", mmlu_image_bytes)
        zip_bytes = zip_buffer.getvalue()
    except Exception:
        zip_bytes = None

    pdf_bytes = None
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import (
            SimpleDocTemplate,
            Paragraph,
            Spacer,
            Image as RLImage,
            PageBreak,
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

        # Key Takeaways
        if key_takeaways:
            elements.append(Spacer(1, 12))
            elements.append(Paragraph("Key Takeaways", heading_style))
            for line in key_takeaways[:6]:
                elements.append(Paragraph(f"• {line}", body_style))

        # Results chart page
        if results_image_bytes is not None:
            elements.append(PageBreak())
            elements.append(Paragraph("Benchmark Results", heading_style))
            elements.append(Spacer(1, 6))
            img_buffer = io.BytesIO(results_image_bytes)
            img = RLImage(img_buffer)
            img._restrictSize(doc.width, doc.height - 100)
            img.hAlign = "CENTER"
            elements.append(img)

        # MMLU chart page
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
    except Exception:
        pdf_bytes = None

    st.download_button(
        label="Download Report as Markdown",
        data=md_content,
        file_name=f"benchmark_report_{timestamp}.md",
        mime="text/markdown",
    )

    if zip_bytes is not None:
        st.download_button(
            label="Download Report as Markdown ZIP (with images)",
            data=zip_bytes,
            file_name=f"benchmark_report_{timestamp}.zip",
            mime="application/zip",
        )
    else:
        st.info(
            "Markdown ZIP export with images requires the 'kaleido' package for Plotly. "
            "Install it with `pip install -U kaleido` and restart the app."
        )

    if pdf_bytes is not None:
        st.download_button(
            label="Download Report as PDF (with images)",
            data=pdf_bytes,
            file_name=f"benchmark_report_{timestamp}.pdf",
            mime="application/pdf",
        )
    else:
        st.info(
            "PDF export requires the 'reportlab' package (and 'kaleido' for chart images). "
            "Install them with `pip install reportlab kaleido` and restart the app."
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
        task_score = task_metrics.get("acc,none")
        if task_score is None:
            task_score = task_metrics.get("acc_norm,none")
        if task_score is None:
            task_score = task_metrics.get("exact_match,none")
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
        "quantization": quantization,
        "temperature": float(temperature),
        "top_p": float(top_p),
        "top_k": int(top_k),
        "repetition_penalty": float(repetition_penalty),
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
                    "--quantization",
                    quantization,
                    "--temperature",
                    str(temperature),
                    "--top_p",
                    str(top_p),
                    "--top_k",
                    str(top_k),
                    "--repetition_penalty",
                    str(repetition_penalty),
                    "--batch_size",
                    str(int(batch_size)),
                    "--output",
                    output_file,
                ]

                if run_deepeval:
                    cmd.append("--deepeval")

                # Run subprocess with real-time logging
                try:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        universal_newlines=True,
                    )

                    full_logs = []
                    latest_line = ""

                    with st.spinner(f"Running **{benchmark}** for **{model}**:"):
                        log_placeholder = st.empty()

                        while True:
                            line = process.stdout.readline()
                            if not line and process.poll() is not None:
                                break
                            if line:
                                full_logs.append(line)
                                candidate = line.rstrip()
                                if candidate:
                                    latest_line = candidate
                                    log_placeholder.code(latest_line, language="bash")

                    # Clear live log line once the run is finished
                    log_placeholder.empty()

                    # Join all logs for download
                    final_logs = "".join(full_logs)

                    if process.returncode != 0:
                        st.error(f"Error running {benchmark} on {model}")
                        if latest_line:
                            st.code(latest_line, language="bash")
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

"""PDF report generation using fpdf2."""

from fpdf import FPDF
from datetime import datetime


def generate_pdf_report(narrative, stats, model_metrics, timestamp=None):
    """Build a formatted PDF report and return as bytes."""
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    # Title page
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 24)
    pdf.cell(0, 20, "Data Analytics", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 12, "Insight Report", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "", 12)
    pdf.cell(0, 10, f"Generated: {timestamp}", new_x="LMARGIN", new_y="NEXT", align="C")
    pdf.cell(0, 10, "AI-Powered Data Analysis Dashboard", new_x="LMARGIN", new_y="NEXT", align="C")

    # Key Statistics
    pdf.ln(15)
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Key Statistics", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)
    pdf.set_font("Helvetica", "", 11)

    stat_items = _build_stat_items(stats)

    for label, value in stat_items:
        pdf.cell(90, 8, f"  {label}:", new_x="RIGHT")
        pdf.cell(0, 8, f"  {value}", new_x="LMARGIN", new_y="NEXT")

    # Model Performance
    if model_metrics:
        pdf.ln(10)
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "Predictive Model Performance", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 11)

        display_metrics = _format_model_metrics(model_metrics)
        for label, value in display_metrics:
            pdf.cell(90, 8, f"  {label}:", new_x="RIGHT")
            pdf.cell(0, 8, f"  {value}", new_x="LMARGIN", new_y="NEXT")

    # AI-Generated Narrative
    if narrative:
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, "AI-Generated Analysis", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "", 10)

        # Handle encoding - replace problematic characters
        clean_text = narrative.encode("latin-1", "replace").decode("latin-1")
        pdf.multi_cell(0, 6, clean_text)

    return bytes(pdf.output())


def _build_stat_items(stats):
    """Build a list of (label, value) pairs from dynamic stats for display."""
    items = []
    items.append(("Total Records", f"{stats.get('total_records', 'N/A'):,}" if isinstance(stats.get('total_records'), (int, float)) else str(stats.get('total_records', 'N/A'))))

    if "unique_entities" in stats:
        items.append(("Unique IDs", f"{stats['unique_entities']:,}" if isinstance(stats['unique_entities'], (int, float)) else str(stats['unique_entities'])))

    if "target_mean" in stats:
        items.append(("Target Mean", f"{stats['target_mean']:.2f}"))
    if "target_median" in stats:
        items.append(("Target Median", f"{stats['target_median']:.2f}"))
    if "target_std" in stats:
        items.append(("Target Std Dev", f"{stats['target_std']:.2f}"))
    if "positive_rate" in stats:
        items.append(("Positive Rate", f"{stats['positive_rate']:.1f}%"))

    # Add correlation stats
    for key, val in stats.items():
        if key.endswith("_target_corr"):
            col_name = key.replace("_target_corr", "").replace("_", " ").title()
            items.append((f"{col_name} Correlation", f"{val:.3f}"))

    # Add category breakdowns summary
    for key, val in stats.items():
        if key.endswith("_breakdown") and isinstance(val, dict):
            cat_name = key.replace("_breakdown", "").replace("_", " ").title()
            if val:
                best = max(val, key=val.get)
                worst = min(val, key=val.get)
                items.append((f"Top {cat_name}", f"{best} ({val[best]:.2f})"))
                items.append((f"Bottom {cat_name}", f"{worst} ({val[worst]:.2f})"))

    return items


def _format_model_metrics(model_metrics):
    """Format model metrics as (label, value) pairs, skipping complex objects."""
    items = []
    skip_keys = {"confusion_matrix", "target_labels"}
    format_as_pct = {"accuracy", "precision", "recall", "f1"}

    for key, value in model_metrics.items():
        if key in skip_keys:
            continue
        label = key.replace("_", " ").title()
        if key in format_as_pct and isinstance(value, (int, float)):
            items.append((label, f"{value:.1%}"))
        else:
            items.append((label, str(value)))
    return items


def build_report_preview(stats, model_metrics, narrative=None):
    """Build markdown preview of what the report contains."""
    sections = []

    # Header
    sections.append("## 📄 Report Preview")
    sections.append(f"*Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}*")
    sections.append("")

    # Key Statistics
    sections.append("### Key Statistics")
    stat_items = _build_stat_items(stats)
    for label, value in stat_items:
        sections.append(f"| **{label}** | {value} |")
    if stat_items:
        # Insert table header
        idx = sections.index("### Key Statistics") + 1
        sections.insert(idx, "| Metric | Value |")
        sections.insert(idx + 1, "|--------|-------|")
    sections.append("")

    # Model Performance
    if model_metrics:
        sections.append("### Predictive Model Performance")
        display_metrics = _format_model_metrics(model_metrics)
        sections.append("| Metric | Value |")
        sections.append("|--------|-------|")
        for label, value in display_metrics:
            sections.append(f"| **{label}** | {value} |")
        sections.append("")

    # AI Narrative
    if narrative:
        sections.append("### AI-Generated Analysis")
        sections.append(narrative)
        sections.append("")

    return "\n".join(sections)

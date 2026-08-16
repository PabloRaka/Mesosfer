#!/usr/bin/env python3
"""
Generate a modern, beautiful benchmark comparison table card (HTML / Image).

Replicates the frontier model launch benchmark style (e.g. Anthropic Opus 4.8 / Mesosfer).
Outputs a self-contained, responsive HTML file that renders beautifully on any screen
and can be saved as an image or embedded into documentation / reports.

Usage:
    # 1. Generate default Mesosfer vs Compact Open-Source LLMs
    python -m scripts.eval.generate_benchmark_card

    # 2. Generate the Frontier Models demo (Opus 4.8 vs GPT-5.5 vs Gemini 3.1 Pro)
    python -m scripts.eval.generate_benchmark_card --preset frontier

    # 3. Custom output file & theme
    python -m scripts.eval.generate_benchmark_card --out benchmark.html --theme peach --open
"""

import json
import argparse
import webbrowser
from pathlib import Path

# ── Preset benchmark data ────────────────────────────────────────────────────

PRESETS = {
    "frontier": {
        "title": "Frontier AI Benchmark Comparison",
        "subtitle": "State-of-the-art evaluation across reasoning, agentic coding, and cybersecurity",
        "highlight_col": 0,
        "theme_color": "#D97745", # Coral/Peach
        "models": [
            {"name": "Opus 4.8", "highlight": True},
            {"name": "Opus 4.7"},
            {"name": "GPT-5.5"},
            {"name": "Gemini 3.1 Pro"},
        ],
        "categories": [
            {
                "title": "Agentic coding",
                "subtitle": "SWE-Bench Pro",
                "values": [
                    {"val": "69.2%", "is_best": True},
                    {"val": "64.3%"},
                    {"val": "58.6%"},
                    {"val": "54.2%"},
                ],
            },
            {
                "title": "Agentic terminal coding",
                "subtitle": "Terminal-Bench 2.1",
                "values": [
                    {"val": "74.6%"},
                    {"val": "66.1%"},
                    {"val": "78.2%", "is_best": True, "highlight_cell": True},
                    {"val": "70.3%"},
                ],
            },
            {
                "title": "Multidisciplinary reasoning",
                "subtitle": "Humanity's Last Exam",
                "split": True,
                "values": [
                    {
                        "top": {"val": "49.8%", "sub": "no tools", "is_best": True},
                        "bot": {"val": "57.9%", "sub": "with tools", "is_best": True},
                    },
                    {
                        "top": {"val": "46.9%", "sub": "no tools"},
                        "bot": {"val": "54.7%", "sub": "with tools"},
                    },
                    {
                        "top": {"val": "41.4%", "sub": "no tools"},
                        "bot": {"val": "52.2%", "sub": "with tools"},
                    },
                    {
                        "top": {"val": "44.4%", "sub": "no tools"},
                        "bot": {"val": "51.4%", "sub": "with tools"},
                    },
                ],
            },
            {
                "title": "Agentic computer use",
                "subtitle": "OSWorld-Verified",
                "values": [
                    {"val": "83.4%", "is_best": True},
                    {"val": "82.8%"},
                    {"val": "78.7%"},
                    {"val": "76.2%"},
                ],
            },
            {
                "title": "Knowledge work",
                "subtitle": "GDPval-AA",
                "values": [
                    {"val": "1890", "is_best": True},
                    {"val": "1753"},
                    {"val": "1769"},
                    {"val": "1314"},
                ],
            },
            {
                "title": "Agentic financial analysis",
                "subtitle": "Finance Agent v2",
                "values": [
                    {"val": "53.9%", "is_best": True},
                    {"val": "51.5%"},
                    {"val": "51.8%"},
                    {"val": "43.0%"},
                ],
            },
        ],
    },
    "mesosfer": {
        "title": "Mesosfer Benchmark Evaluation",
        "subtitle": "Lightweight Domain-Specialized LLM for Cybersecurity & Defensive Reasoning",
        "highlight_col": 0,
        "theme_color": "#0ea5e9", # Sky/Cyan
        "models": [
            {"name": "Mesosfer-d12 (380M)", "highlight": True},
            {"name": "SmolLM2-360M"},
            {"name": "Qwen2.5-0.5B"},
            {"name": "Llama-3.2-1B"},
        ],
        "categories": [
            {
                "title": "Cybersecurity Vulnerability Triage",
                "subtitle": "SecBench (CWE/CVE)",
                "values": [
                    {"val": "68.4%", "is_best": True},
                    {"val": "41.2%"},
                    {"val": "49.8%"},
                    {"val": "53.1%"},
                ],
            },
            {
                "title": "Defensive Security QA",
                "subtitle": "CyberMetric-80",
                "values": [
                    {"val": "64.8%", "is_best": True},
                    {"val": "36.5%"},
                    {"val": "45.0%"},
                    {"val": "48.2%"},
                ],
            },
            {
                "title": "Multidisciplinary Knowledge",
                "subtitle": "MMLU Benchmark (57 Subjects)",
                "values": [
                    {"val": "32.4%"},
                    {"val": "28.5%"},
                    {"val": "34.1%", "is_best": True, "highlight_cell": True},
                    {"val": "39.2%", "is_best": True, "highlight_cell": True},
                ],
            },
            {
                "title": "Scientific & Common Reasoning",
                "subtitle": "ARC-Challenge & ARC-Easy",
                "split": True,
                "values": [
                    {
                        "top": {"val": "31.2%", "sub": "ARC-Challenge"},
                        "bot": {"val": "35.8%", "sub": "ARC-Easy", "is_best": True},
                    },
                    {
                        "top": {"val": "26.4%", "sub": "ARC-Challenge"},
                        "bot": {"val": "30.1%", "sub": "ARC-Easy"},
                    },
                    {
                        "top": {"val": "33.5%", "sub": "ARC-Challenge", "is_best": True, "highlight_cell": True},
                        "bot": {"val": "34.2%", "sub": "ARC-Easy"},
                    },
                    {
                        "top": {"val": "36.8%", "sub": "ARC-Challenge", "is_best": True, "highlight_cell": True},
                        "bot": {"val": "38.9%", "sub": "ARC-Easy", "is_best": True, "highlight_cell": True},
                    },
                ],
            },
            {
                "title": "Code Generation & Execution",
                "subtitle": "HumanEval (Python pass@1)",
                "values": [
                    {"val": "16.8%"},
                    {"val": "10.4%"},
                    {"val": "18.2%", "is_best": True, "highlight_cell": True},
                    {"val": "23.5%", "is_best": True, "highlight_cell": True},
                ],
            },
            {
                "title": "Indonesian Instruction Following",
                "subtitle": "IndoMMLU & Alpaca-ID",
                "values": [
                    {"val": "71.5%", "is_best": True},
                    {"val": "38.2%"},
                    {"val": "48.9%"},
                    {"val": "52.4%"},
                ],
            },
        ],
    },
}

# ── HTML / CSS Template ──────────────────────────────────────────────────────

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{page_title}</title>
  <!-- Google Fonts: Inter & Newsreader for editorial look -->
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&display=swap" rel="stylesheet">
  
  <style>
    :root {{
      --primary-color: {theme_color};
      --primary-tint: {theme_tint};
      --bg-page: #FBFBFC;
      --card-bg: #FFFFFF;
      --text-main: #18181B;
      --text-muted: #71717A;
      --text-submuted: #A1A1AA;
      --border-color: #F1F1F4;
      --cell-highlight: #FBF7EE;
      --cell-highlight-border: #EFE6D5;
      --shadow-sm: 0 4px 20px -2px rgba(0, 0, 0, 0.05);
      --shadow-lg: 0 20px 40px -15px rgba(0, 0, 0, 0.07);
    }}

    * {{
      box-sizing: border-box;
      margin: 0;
      padding: 0;
    }}

    body {{
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background-color: var(--bg-page);
      color: var(--text-main);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      padding: 40px 20px;
    }}

    .container {{
      max-width: 1120px;
      width: 100%;
      margin: 0 auto;
    }}

    .header-area {{
      margin-bottom: 24px;
      text-align: center;
    }}

    .header-area h1 {{
      font-family: 'Newsreader', serif;
      font-size: 2.2rem;
      font-weight: 500;
      letter-spacing: -0.02em;
      color: var(--text-main);
      margin-bottom: 8px;
    }}

    .header-area p {{
      color: var(--text-muted);
      font-size: 0.95rem;
    }}

    /* Main Card */
    .benchmark-card {{
      background: var(--card-bg);
      border-radius: 24px;
      border: 1px solid var(--border-color);
      box-shadow: var(--shadow-lg);
      padding: 32px 28px;
      overflow-x: auto;
      position: relative;
    }}

    table.benchmark-table {{
      width: 100%;
      border-collapse: separate;
      border-spacing: 0;
      position: relative;
    }}

    /* Header formatting */
    th {{
      font-family: 'Newsreader', serif;
      font-size: 1.35rem;
      font-weight: 500;
      color: var(--text-main);
      padding: 20px 16px;
      text-align: center;
      vertical-align: middle;
      border-bottom: 1px solid var(--border-color);
    }}

    th.col-category {{
      text-align: left;
      font-family: 'Inter', sans-serif;
      font-size: 0.9rem;
      font-weight: 600;
      color: var(--text-muted);
      width: 28%;
      padding-left: 20px;
    }}

    /* Highlighted Model Column */
    .highlight-pill {{
      position: absolute;
      top: 10px;
      bottom: 10px;
      left: {pill_left_pct}%;
      width: {pill_width_pct}%;
      background: var(--primary-tint);
      border: 2px solid var(--primary-color);
      border-radius: 20px;
      pointer-events: none;
      z-index: 1;
      transition: all 0.2s ease;
    }}

    /* Rows & Cells */
    tr {{
      position: relative;
    }}

    td {{
      padding: 24px 16px;
      text-align: center;
      vertical-align: middle;
      border-bottom: 1px solid var(--border-color);
      position: relative;
      z-index: 2;
    }}

    tr:last-child td {{
      border-bottom: none;
    }}

    td.category-cell {{
      text-align: left;
      padding-left: 20px;
    }}

    .category-title {{
      font-weight: 600;
      font-size: 1.02rem;
      color: var(--text-main);
      margin-bottom: 3px;
      line-height: 1.3;
    }}

    .category-subtitle {{
      font-size: 0.82rem;
      color: var(--text-muted);
      font-weight: 400;
    }}

    /* Value Display */
    .val-single {{
      font-weight: 700;
      font-size: 1.25rem;
      letter-spacing: -0.01em;
      color: var(--text-main);
      display: inline-block;
      padding: 6px 12px;
      border-radius: 8px;
    }}

    .val-single.cell-boxed {{
      background: var(--cell-highlight);
      border: 1px solid var(--cell-highlight-border);
    }}

    /* Split Values (e.g. no tools / with tools) */
    .split-stack {{
      display: flex;
      flex-direction: column;
      align-items: center;
      gap: 12px;
    }}

    .split-subitem {{
      display: flex;
      flex-direction: column;
      align-items: center;
    }}

    .split-val {{
      font-weight: 700;
      font-size: 1.15rem;
      letter-spacing: -0.01em;
      color: var(--text-main);
    }}

    .split-label {{
      font-size: 0.72rem;
      color: var(--text-muted);
      font-weight: 500;
      margin-top: 1px;
    }}

    .split-divider {{
      width: 48px;
      height: 1px;
      background: rgba(0, 0, 0, 0.08);
      margin: 2px 0;
    }}

    /* Footer badge */
    .card-footer {{
      margin-top: 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      font-size: 0.8rem;
      color: var(--text-submuted);
      padding: 0 10px;
    }}

    .badge-pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 10px;
      background: rgba(0, 0, 0, 0.04);
      border-radius: 12px;
      font-weight: 500;
      color: var(--text-muted);
    }}
  </style>
</head>
<body>

  <div class="container">
    <div class="header-area">
      <h1>{title}</h1>
      <p>{subtitle}</p>
    </div>

    <div class="benchmark-card">
      <table class="benchmark-table">
        <div class="highlight-pill"></div>
        <thead>
          <tr>
            <th class="col-category"></th>
            {header_th_html}
          </tr>
        </thead>
        <tbody>
          {rows_tbody_html}
        </tbody>
      </table>
    </div>

    <div class="card-footer">
      <div class="badge-pill">⚡ Mesosfer Frontier AI Architecture</div>
      <div>Evaluated on standard benchmarks</div>
    </div>
  </div>

</body>
</html>
"""

def generate_html(data: dict, theme_color_override: str | None = None) -> str:
    """Generate the full HTML string from a benchmark data dictionary."""
    models = data["models"]
    categories = data["categories"]
    num_models = len(models)
    highlight_idx = data.get("highlight_col", 0)

    # Theme colors
    theme_color = theme_color_override or data.get("theme_color", "#D97745")
    # Generate soft tint (rgba)
    if theme_color.startswith("#") and len(theme_color) == 7:
        r = int(theme_color[1:3], 16)
        g = int(theme_color[3:5], 16)
        b = int(theme_color[5:7], 16)
        theme_tint = f"rgba({r}, {g}, {b}, 0.09)"
    else:
        theme_tint = "rgba(217, 119, 69, 0.09)"

    # Calculate highlight pill position (table percentage)
    # Category col = 28%, remaining = 72%
    cat_width = 28.0
    model_width = (100.0 - cat_width) / num_models
    pill_left_pct = cat_width + (highlight_idx * model_width) + 0.5
    pill_width_pct = model_width - 1.0

    # Build Header <th>
    th_list = []
    for i, m in enumerate(models):
        th_list.append(f'<th class="model-th">{m["name"]}</th>')
    header_th_html = "\n            ".join(th_list)

    # Build Rows <tbody>
    rows_list = []
    for cat in categories:
        cat_title = cat["title"]
        cat_sub = cat.get("subtitle", "")
        is_split = cat.get("split", False)
        values = cat["values"]

        td_cells = []
        for i, val_entry in enumerate(values):
            if is_split:
                top = val_entry["top"]
                bot = val_entry["bot"]
                cell_html = f"""
                <div class="split-stack">
                  <div class="split-subitem">
                    <span class="split-val">{top['val']}</span>
                    <span class="split-label">{top['sub']}</span>
                  </div>
                  <div class="split-divider"></div>
                  <div class="split-subitem">
                    <span class="split-val">{bot['val']}</span>
                    <span class="split-label">{bot['sub']}</span>
                  </div>
                </div>
                """
            else:
                v = val_entry["val"]
                boxed_class = " cell-boxed" if val_entry.get("highlight_cell") else ""
                cell_html = f'<span class="val-single{boxed_class}">{v}</span>'

            td_cells.append(f'<td>{cell_html}</td>')

        row_html = f"""
          <tr>
            <td class="category-cell">
              <div class="category-title">{cat_title}</div>
              <div class="category-subtitle">{cat_sub}</div>
            </td>
            {''.join(td_cells)}
          </tr>
        """
        rows_list.append(row_html)

    rows_tbody_html = "\n".join(rows_list)

    return HTML_TEMPLATE.format(
        page_title=data.get("title", "Benchmark Comparison"),
        title=data.get("title", "Benchmark Comparison"),
        subtitle=data.get("subtitle", ""),
        theme_color=theme_color,
        theme_tint=theme_tint,
        pill_left_pct=f"{pill_left_pct:.2f}",
        pill_width_pct=f"{pill_width_pct:.2f}",
        header_th_html=header_th_html,
        rows_tbody_html=rows_tbody_html,
    )


# ── Main Entrypoint ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate modern benchmark comparison table cards.")
    parser.add_argument("--preset", choices=list(PRESETS.keys()), default="frontier",
                        help="Benchmark preset template (frontier or mesosfer)")
    parser.add_argument("--json", type=str, default=None,
                        help="Path to custom JSON file containing benchmark data")
    parser.add_argument("--out", type=str, default="benchmark_card.html",
                        help="Output HTML file path (default: benchmark_card.html)")
    parser.add_argument("--theme", type=str, default=None,
                        help="Theme color override (e.g. '#D97745', '#0ea5e9', '#10b981')")
    parser.add_argument("--open", action="store_true", default=False,
                        help="Automatically open generated HTML in default web browser")
    args = parser.parse_args()

    if args.json:
        with open(args.json, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = PRESETS[args.preset]

    html_content = generate_html(data, theme_color_override=args.theme)

    out_path = Path(args.out).resolve()
    out_path.write_text(html_content, encoding="utf-8")
    print(f"\n[OK] Benchmark card generated successfully: {out_path}")

    if args.open:
        try:
            webbrowser.open(out_path.as_uri())
        except Exception:
            pass


if __name__ == "__main__":
    main()

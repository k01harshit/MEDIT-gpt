import re
import html
from typing import List, Dict, Any

# Vibrant, accessible color palette for distinguishing different papers
PAPER_PALETTE = [
    {
        "name": "Sky Blue",
        "text": "#38BDF8",
        "bg": "rgba(56, 189, 248, 0.16)",
        "border": "#0284C7",
        "border_subtle": "rgba(56, 189, 248, 0.40)",
    },
    {
        "name": "Emerald Green",
        "text": "#34D399",
        "bg": "rgba(52, 211, 153, 0.16)",
        "border": "#059669",
        "border_subtle": "rgba(52, 211, 153, 0.40)",
    },
    {
        "name": "Coral Red",
        "text": "#F87171",
        "bg": "rgba(248, 113, 113, 0.16)",
        "border": "#DC2626",
        "border_subtle": "rgba(248, 113, 113, 0.40)",
    },
    {
        "name": "Amber Yellow",
        "text": "#FBBF24",
        "bg": "rgba(251, 191, 36, 0.16)",
        "border": "#D97706",
        "border_subtle": "rgba(251, 191, 36, 0.40)",
    },
    {
        "name": "Purple Violet",
        "text": "#C084FC",
        "bg": "rgba(192, 132, 252, 0.16)",
        "border": "#7C3AED",
        "border_subtle": "rgba(192, 132, 252, 0.40)",
    },
    {
        "name": "Cyan",
        "text": "#22D3EE",
        "bg": "rgba(34, 211, 238, 0.16)",
        "border": "#0891B2",
        "border_subtle": "rgba(34, 211, 238, 0.40)",
    },
    {
        "name": "Rose Pink",
        "text": "#FB7185",
        "bg": "rgba(251, 113, 133, 0.16)",
        "border": "#E11D48",
        "border_subtle": "rgba(251, 113, 133, 0.40)",
    },
]


def extract_exact_paragraph(raw_text: str) -> str:
    """Extracts the exact paragraph text written in the paper without internal index headers."""
    if not raw_text:
        return ""
    # Chunks are typically indexed as "[Paper Title | Section Heading]\nActual paragraph text..."
    if raw_text.startswith("[") and "]\n" in raw_text:
        return raw_text.split("]\n", 1)[1].strip()
    return raw_text.strip()


def make_paper_badge(title: str, color_dict: dict) -> str:
    """Generates an HTML inline pill badge with the paper name and distinct color styling."""
    escaped_title = html.escape(title)
    return (
        f'<span class="paper-citation-badge" style="'
        f'color: {color_dict["text"]} !important; '
        f'background-color: {color_dict["bg"]}; '
        f'border: 1px solid {color_dict["border_subtle"]}; '
        f'border-radius: 6px; '
        f'padding: 2px 8px; '
        f'font-size: 0.83em; '
        f'font-weight: 700; '
        f'display: inline-block; '
        f'margin: 1px 3px; '
        f'vertical-align: baseline;" '
        f'title="{escaped_title}">'
        f'📄 {escaped_title}</span>'
    )


def md_to_html(text: str) -> str:
    """Safely converts basic markdown bold/italics and paragraphs into clean HTML."""
    # Convert bold **text** to <strong>text</strong>
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    # Convert italic *text* or _text_ to <em>text</em>
    text = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", text)

    paras = text.strip().split("\n\n")
    out = []
    for p in paras:
        p = p.strip()
        if not p:
            continue
        # Handle bullet points
        if p.startswith("- ") or p.startswith("* "):
            lines = p.split("\n")
            items = []
            for line in lines:
                line = line.strip()
                if line.startswith("- ") or line.startswith("* "):
                    items.append(f"<li style='margin-bottom: 4px;'>{line[2:].strip()}</li>")
                elif line:
                    items.append(f"<p style='margin-bottom: 6px;'>{line}</p>")
            out.append(f"<ul style='margin: 8px 0 12px 20px; padding-left: 0;'>{''.join(items)}</ul>")
        else:
            out.append(f"<p style='margin-bottom: 10px; line-height: 1.68;'>{p.replace(chr(10), '<br/>')}</p>")
    return "".join(out)


def format_rag_response(raw_llm_answer: str, context_chunks: List[Dict[str, Any]]) -> str:
    """Formats the final response into two visually distinct sections:
    1. Exact Paragraph(s) Written in Paper (Emerald green theme, with paper name in unique colors).
    2. Summarized Answer through RAG (Indigo/Blue theme, with in-text paper title badges instead of PMIDs).
    """
    if not context_chunks:
        return raw_llm_answer

    # Check for insufficient evidence response
    if "insufficient evidence" in raw_llm_answer.lower():
        return (
            '<div class="rag-summary-container" style="'
            'background: rgba(99, 102, 241, 0.08); '
            'border: 1.5px solid rgba(99, 102, 241, 0.35); '
            'border-radius: 16px; padding: 18px 20px;">'
            '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 12px;">'
            '<span style="background: linear-gradient(135deg, #4285F4, #6366F1); color: #FFFFFF !important; font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 999px;">RAG SYNTHESIS</span>'
            '<span style="color: #818CF8 !important; font-size: 1.05rem; font-weight: 700;">Summarized Answer</span>'
            '</div>'
            '<p style="margin: 0; line-height: 1.6;">Insufficient evidence in the indexed literature to answer this.</p>'
            '</div>'
        )

    # -----------------------------------------------------
    # 1. Build mappings for unique papers (Titles, PMIDs, Colors)
    # -----------------------------------------------------
    paper_color_map = {}
    paper_by_pmid = {}
    paper_by_idx = {}
    ordered_papers = []

    for idx, chunk in enumerate(context_chunks, 1):
        title = chunk.get("title", "").strip() or f"Paper #{idx}"
        pmid = str(chunk.get("pmid", "")).strip()

        if title not in paper_color_map:
            color_idx = len(paper_color_map) % len(PAPER_PALETTE)
            paper_color_map[title] = PAPER_PALETTE[color_idx]
            ordered_papers.append(title)

        paper_by_idx[idx] = title
        if pmid:
            paper_by_pmid[pmid] = title

    # -----------------------------------------------------
    # 2. Section 1: Exact Paragraph(s) Written in Paper
    # -----------------------------------------------------
    chunk_cards_html = []
    seen_paras = set()

    # Show up to top 3 distinct evidence paragraphs
    for chunk in context_chunks[:3]:
        title = chunk.get("title", "").strip() or "Untitled Paper"
        color = paper_color_map.get(title, PAPER_PALETTE[0])
        section = chunk.get("section", "Passage")
        year = chunk.get("year", "")
        exact_text = extract_exact_paragraph(chunk.get("text", ""))

        if exact_text in seen_paras:
            continue
        seen_paras.add(exact_text)

        year_str = f" &bull; {year}" if year else ""
        escaped_title = html.escape(title)
        escaped_para = html.escape(exact_text)

        card = (
            f'<div class="rag-chunk-card" style="'
            f'background: rgba(0, 0, 0, 0.22); '
            f'border: 1px solid rgba(255, 255, 255, 0.08); '
            f'border-left: 4.5px solid {color["text"]}; '
            f'border-radius: 12px; '
            f'padding: 14px 16px; '
            f'margin-bottom: 12px;">'
            f'<div style="display: flex; flex-wrap: wrap; align-items: center; justify-content: space-between; gap: 8px; margin-bottom: 8px;">'
            f'<span style="color: {color["text"]} !important; font-weight: 700; font-size: 0.95rem;">'
            f'📄 {escaped_title}</span>'
            f'<span style="background: rgba(255, 255, 255, 0.08); font-size: 0.78rem; padding: 2px 8px; border-radius: 4px;">'
            f'Section: <b>{html.escape(section)}</b>{year_str}</span>'
            f'</div>'
            f'<div style="font-size: 0.92rem; line-height: 1.62; border-top: 1px dashed rgba(255, 255, 255, 0.1); padding-top: 8px;">'
            f'"{escaped_para}"</div>'
            f'</div>'
        )
        chunk_cards_html.append(card)

    exact_section_html = (
        '<div class="rag-exact-container" style="'
        'background: rgba(16, 185, 129, 0.08); '
        'border: 1.5px solid rgba(16, 185, 129, 0.35); '
        'border-radius: 18px; padding: 18px 22px; margin-bottom: 18px;">'
        '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">'
        '<span style="background: #10B981; color: #FFFFFF !important; font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em;">Source Evidence</span>'
        '<span style="color: #34D399 !important; font-size: 1.05rem; font-weight: 700;">Exact Paragraph(s) Written in Paper</span>'
        '</div>'
        + "".join(chunk_cards_html) +
        '</div>'
    )

    # -----------------------------------------------------
    # 3. Section 2: Summarized Answer through RAG
    # -----------------------------------------------------
    processed_summary = raw_llm_answer

    # Replace bracketed citations: [PMID 42727510], [Paper #1], [Paper: Title], or [Title]
    def replace_bracket_content(match):
        inner = match.group(1).strip()

        # Check for numeric PMIDs inside brackets (e.g., [PMID 42727510, 37386505])
        pmid_matches = re.findall(r"\b(\d{6,9})\b", inner)
        if pmid_matches:
            badges = []
            for pmid_val in pmid_matches:
                if pmid_val in paper_by_pmid:
                    t = paper_by_pmid[pmid_val]
                    badges.append(make_paper_badge(t, paper_color_map[t]))
                elif ordered_papers:
                    t = ordered_papers[0]
                    badges.append(make_paper_badge(t, paper_color_map[t]))
            if badges:
                return " ".join(badges)

        # Check for Paper #idx (e.g., [Paper #1], [Paper 2])
        paper_idx_match = re.search(r"Paper\s*#?(\d+)", inner, re.IGNORECASE)
        if paper_idx_match:
            idx_val = int(paper_idx_match.group(1))
            if idx_val in paper_by_idx:
                t = paper_by_idx[idx_val]
                return make_paper_badge(t, paper_color_map[t])

        # Check for Paper Title match
        clean_cand = re.sub(r"^(?:Paper:\s*|Title:\s*)", "", inner, flags=re.IGNORECASE).strip().strip('"').strip("'")
        for t in ordered_papers:
            if clean_cand.lower() == t.lower() or clean_cand.lower() in t.lower() or t.lower() in clean_cand.lower():
                return make_paper_badge(t, paper_color_map[t])

        # Fallback if it contains PMID keyword
        if "PMID" in inner.upper():
            return ""

        return match.group(0)

    # Replace all [ ... ] brackets
    processed_summary = re.sub(r"\[([^\]]+)\]", replace_bracket_content, processed_summary)

    # Replace any leftover raw "PMID 42727510" in prose
    for pmid, t in paper_by_pmid.items():
        processed_summary = re.sub(
            rf"\bPMID\s*{pmid}\b",
            make_paper_badge(t, paper_color_map[t]),
            processed_summary,
            flags=re.IGNORECASE,
        )

    # Remove any stray unclosed or leftover PMID mentions
    processed_summary = re.sub(r"\[?PMID\s*\d{6,9}\]?", "", processed_summary, flags=re.IGNORECASE)

    html_summary = md_to_html(processed_summary)

    summary_section_html = (
        '<div class="rag-summary-container" style="'
        'background: rgba(99, 102, 241, 0.08); '
        'border: 1.5px solid rgba(99, 102, 241, 0.35); '
        'border-radius: 18px; padding: 18px 22px;">'
        '<div style="display: flex; align-items: center; gap: 8px; margin-bottom: 14px;">'
        '<span style="background: linear-gradient(135deg, #4285F4, #6366F1); color: #FFFFFF !important; font-size: 0.75rem; font-weight: 800; padding: 3px 10px; border-radius: 999px; text-transform: uppercase; letter-spacing: 0.05em;">RAG Synthesis</span>'
        '<span style="color: #818CF8 !important; font-size: 1.05rem; font-weight: 700;">Summarized Answer</span>'
        '</div>'
        f'<div style="font-size: 0.96rem; line-height: 1.7;">{html_summary}</div>'
        '</div>'
    )

    return f"{exact_section_html}\n\n{summary_section_html}"

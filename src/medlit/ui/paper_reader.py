import os
import io
import html
import json
import requests
import pandas as pd
import pymupdf
import streamlit as st
from lxml import etree

FULLTEXT_JSONL = "data/processed/parkinsons_fulltext.jsonl"
ABSTRACTS_JSONL = "data/processed/parkinsons_abstracts_meta.jsonl"
PDF_DIR = "data/raw/pdfs"
ANNOTATED_DIR = "data/annotated_pdfs"
PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"

HIGHLIGHT_COLORS = {
    "Yellow": (1.0, 0.92, 0.23),
    "Green": (0.51, 0.79, 0.58),
    "Pink": (0.95, 0.55, 0.51),
    "Blue": (0.54, 0.71, 0.97),
}


def _parse_year(val) -> int:
    try:
        return int(str(val)[:4])
    except (ValueError, TypeError):
        return 2024


@st.cache_data
def load_paper_catalog() -> pd.DataFrame:
    """Merge full-text Open Access papers and PubMed abstracts into one searchable catalog."""
    records = {}

    if os.path.exists(ABSTRACTS_JSONL):
        with open(ABSTRACTS_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pmid = str(row.get("pmid", "")).strip()
                if not pmid:
                    continue
                records[pmid] = {
                    "pmid": pmid,
                    "pmcid": "",
                    "title": row.get("title", "Untitled"),
                    "journal": row.get("journal") or "PubMed Indexed",
                    "year": _parse_year(row.get("year")),
                    "abstract": row.get("abstract", ""),
                    "sections": [],
                    "open_access": False,
                }

    if os.path.exists(FULLTEXT_JSONL):
        with open(FULLTEXT_JSONL, "r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                pmid = str(row.get("pmid") or row.get("pmcid", "")).strip()
                if not pmid:
                    continue
                existing = records.get(pmid, {})
                records[pmid] = {
                    "pmid": pmid,
                    "pmcid": row.get("pmcid", ""),
                    "title": row.get("title") or existing.get("title", "Untitled"),
                    "journal": row.get("journal") or existing.get("journal") or "PMC Open Access",
                    "year": _parse_year(row.get("year") or existing.get("year")),
                    "abstract": row.get("abstract") or existing.get("abstract", ""),
                    "sections": row.get("sections", []),
                    "open_access": True,
                }

    return pd.DataFrame(list(records.values())) if records else pd.DataFrame()


def fetch_or_build_pdf_bytes(paper: dict) -> bytes | None:
    """Return original PDF bytes if available locally/via PMC OA, or compile full-text via DocumentWriter."""
    os.makedirs(PDF_DIR, exist_ok=True)
    pmcid = paper.get("pmcid", "")

    # 1. Check local cached PDF
    if pmcid:
        local_path = os.path.join(PDF_DIR, f"{pmcid}.pdf")
        if os.path.exists(local_path):
            with open(local_path, "rb") as f:
                return f.read()

        # 2. Try fetching original Open Access PDF from NCBI PMC OA API
        try:
            resp = requests.get(PMC_OA_API, params={"id": pmcid}, timeout=12)
            if resp.status_code == 200:
                root = etree.fromstring(resp.content)
                links = root.xpath("//link[@format='pdf']/@href")
                if links:
                    pdf_url = links[0].replace("ftp://", "https://")
                    pdf_resp = requests.get(pdf_url, timeout=25)
                    if pdf_resp.status_code == 200 and pdf_resp.content[:4] == b"%PDF":
                        with open(local_path, "wb") as f:
                            f.write(pdf_resp.content)
                        return pdf_resp.content
        except Exception:
            pass

    # 3. Compile Open Access XML full-text into a clean in-memory PDF using DocumentWriter
    if paper.get("open_access") and (paper.get("sections") or paper.get("abstract")):
        title_esc = html.escape(str(paper.get("title", "Untitled")))
        venue_esc = html.escape(str(paper.get("journal", "PMC Open Access")))
        pmid_esc = html.escape(str(paper.get("pmid", "N/A")))
        pmcid_esc = html.escape(str(pmcid or "N/A"))
        year_esc = html.escape(str(paper.get("year", "")))

        html_parts = [
            f"<h2>{title_esc}</h2>",
            f"<p><b>Venue:</b> {venue_esc} | <b>Year:</b> {year_esc} | <b>PMID:</b> {pmid_esc} | <b>PMCID:</b> {pmcid_esc}</p><hr>",
        ]
        if paper.get("abstract"):
            abs_esc = html.escape(str(paper["abstract"]))
            html_parts.append(f"<h3>Abstract</h3><p>{abs_esc}</p>")
        for sec in paper.get("sections", []):
            h_esc = html.escape(str(sec.get("heading", "Section")))
            t_esc = html.escape(str(sec.get("text", ""))).replace("\n", "<br>")
            html_parts.append(f"<h3>{h_esc}</h3><p>{t_esc}</p>")

        story = pymupdf.Story(html="\n".join(html_parts))
        mediabox = pymupdf.paper_rect("a4")
        where = mediabox + (42, 42, -42, -42)

        buf = io.BytesIO()
        writer = pymupdf.DocumentWriter(buf)
        more = True
        while more:
            dev = writer.begin_page(mediabox)
            more, _ = story.place(where)
            story.draw(dev)
            writer.end_page()
        writer.close()
        return buf.getvalue()

    return None


def apply_pdf_annotations(pdf_bytes: bytes, highlights: list, notes: list) -> bytes:
    """Apply PyMuPDF text highlights and sticky notes to the PDF and return updated bytes."""
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

    for h in highlights:
        needle = h["text"].strip()
        color = HIGHLIGHT_COLORS.get(h["color"], (1.0, 0.92, 0.23))
        if not needle:
            continue
        for page in doc:
            quads = page.search_for(needle, quads=True)
            for q in quads:
                annot = page.add_highlight_annot(q)
                annot.set_colors(stroke=color)
                annot.update()

    for n in notes:
        p_idx = max(0, min(n["page"] - 1, len(doc) - 1))
        page = doc[p_idx]
        point = pymupdf.Point(45, 55 + (n.get("offset", 0) * 24))
        page.add_text_annot(point, n["text"], icon="Note")

    updated = doc.tobytes(deflate=True)
    doc.close()
    return updated


def _render_reader_view(paper: dict):
    """Render the native PyMuPDF Page Viewer & Annotation Studio (or Abstract fallback)."""
    selected_pmid = paper["pmid"]
    is_marked = selected_pmid in st.session_state.bookmarked_pmids

    top_c1, _ = st.columns([1.4, 4.6])
    with top_c1:
        if st.button("Back to Search Results", use_container_width=True, type="secondary"):
            st.session_state.selected_paper_pmid = None
            st.rerun()

    st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)

    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.markdown(f"### {paper['title']}")
        st.caption(f"Venue: {paper['journal']}  |  Year: {paper['year']}  |  PMID: {paper['pmid']}")
    with h_col2:
        mark_label = "Marked / Bookmarked" if is_marked else "Mark Paper"
        if st.button(mark_label, use_container_width=True, type="primary" if is_marked else "secondary"):
            if is_marked:
                st.session_state.bookmarked_pmids.remove(selected_pmid)
            else:
                st.session_state.bookmarked_pmids.add(selected_pmid)
            st.rerun()

    st.divider()

    # Non-Open Access Flow
    if not paper["open_access"]:
        st.warning("This paper is not Open Access (full PDF is unavailable). Showing the abstract below.")
        st.markdown("#### Abstract")
        st.write(paper["abstract"] or "No abstract text available.")
        st.text_area("Personal Notes on this Paper", key=f"abs_note_{selected_pmid}", height=140)
        return

    # Open Access Full PDF Reader + Annotation Studio
    hl_key = f"hl_{selected_pmid}"
    nt_key = f"nt_{selected_pmid}"
    if hl_key not in st.session_state:
        st.session_state[hl_key] = []
    if nt_key not in st.session_state:
        st.session_state[nt_key] = []

    with st.spinner("Loading PDF document..."):
        base_pdf = fetch_or_build_pdf_bytes(paper)

    if not base_pdf:
        st.warning("Full PDF could not be retrieved. Showing the abstract instead.")
        st.write(paper["abstract"] or "No abstract text available.")
        return

    annotated_pdf = apply_pdf_annotations(
        base_pdf, st.session_state[hl_key], st.session_state[nt_key]
    )

    reader_col, tools_col = st.columns([2.2, 1], gap="large")

    with reader_col:
        doc_view = pymupdf.open(stream=annotated_pdf, filetype="pdf")
        total_pages = len(doc_view)

        pg_key = f"view_page_{selected_pmid}"
        if pg_key not in st.session_state:
            st.session_state[pg_key] = 1
        if st.session_state[pg_key] > total_pages:
            st.session_state[pg_key] = 1

        p_col1, p_col2, p_col3 = st.columns([1, 2, 1])
        with p_col1:
            if st.button(
                "Previous Page",
                key=f"prev_{selected_pmid}",
                use_container_width=True,
                disabled=st.session_state[pg_key] <= 1,
            ):
                st.session_state[pg_key] -= 1
                st.rerun()
        with p_col2:
            st.markdown(
                f"<div style='text-align:center; padding-top:8px; font-weight:700;'>"
                f"Page {st.session_state[pg_key]} of {total_pages}</div>",
                unsafe_allow_html=True,
            )
        with p_col3:
            if st.button(
                "Next Page",
                key=f"next_{selected_pmid}",
                use_container_width=True,
                disabled=st.session_state[pg_key] >= total_pages,
            ):
                st.session_state[pg_key] += 1
                st.rerun()

        page = doc_view[st.session_state[pg_key] - 1]
        pix = page.get_pixmap(dpi=150, annots=True)
        st.image(pix.tobytes("png"), use_container_width=True)
        doc_view.close()

    with tools_col:
        st.markdown("#### Annotation Studio")

        hl_text = st.text_input("Text to highlight in PDF", key=f"in_hl_{selected_pmid}")
        hl_color = st.selectbox("Highlight Color", list(HIGHLIGHT_COLORS.keys()), key=f"col_hl_{selected_pmid}")
        if st.button("Add Highlight", use_container_width=True):
            if hl_text.strip():
                st.session_state[hl_key].append({"text": hl_text.strip(), "color": hl_color})
                st.rerun()

        st.divider()

        note_page = st.number_input("Page Number", min_value=1, value=st.session_state.get(pg_key, 1), step=1, key=f"pg_nt_{selected_pmid}")
        note_text = st.text_area("Write a note over this page", height=90, key=f"in_nt_{selected_pmid}")
        if st.button("Attach Note to Page", use_container_width=True):
            if note_text.strip():
                st.session_state[nt_key].append({
                    "page": int(note_page),
                    "text": note_text.strip(),
                    "offset": len(st.session_state[nt_key]),
                })
                st.rerun()

        if st.session_state[hl_key] or st.session_state[nt_key]:
            st.caption(
                f"Active: {len(st.session_state[hl_key])} highlight(s), "
                f"{len(st.session_state[nt_key])} note(s)"
            )

        st.divider()

        filename = f"{paper['pmcid'] or paper['pmid']}_annotated.pdf"
        st.download_button(
            "Download Edited PDF",
            data=annotated_pdf,
            file_name=filename,
            mime="application/pdf",
            use_container_width=True,
        )
        if st.button("Save Edited Paper Locally", use_container_width=True):
            os.makedirs(ANNOTATED_DIR, exist_ok=True)
            out_path = os.path.join(ANNOTATED_DIR, filename)
            with open(out_path, "wb") as f:
                f.write(annotated_pdf)
            st.success(f"Saved to {out_path}")


def render_paper_explorer():
    """Main UI for Search & Read Papers (paginated full list view, no dropdown)."""
    df = load_paper_catalog()
    if df.empty:
        st.warning("No paper datasets found. Run the ingestion scripts first.")
        return

    if "bookmarked_pmids" not in st.session_state:
        st.session_state.bookmarked_pmids = set()
    if "selected_paper_pmid" not in st.session_state:
        st.session_state.selected_paper_pmid = None

    # Open the Reader View if a paper was clicked
    if st.session_state.selected_paper_pmid is not None:
        match = df[df["pmid"] == st.session_state.selected_paper_pmid]
        if not match.empty:
            _render_reader_view(match.iloc[0].to_dict())
            return
        st.session_state.selected_paper_pmid = None

    # 1. SEARCH & FILTER CONTROLS
    st.subheader("Search & Read Parkinson's Papers")
    search_mode = st.radio(
        "Search By",
        ["All Fields", "Keywords", "Paper Name", "Journal / Venue"],
        horizontal=True,
    )

    f_col1, f_col2, f_col3, f_col4 = st.columns([2.2, 1.2, 1.3, 1.0])
    with f_col1:
        query = st.text_input(
            "Search query",
            placeholder="Type keywords, paper title, or journal name...",
            label_visibility="collapsed",
        )
    with f_col2:
        min_y, max_y = int(df["year"].min()), int(df["year"].max())
        yr_filter = st.slider("Year Range", min_y, max_y, (min_y, max_y))
    with f_col3:
        venues = sorted(df["journal"].dropna().unique().tolist())
        selected_venues = st.multiselect("Filter by Venue / Journal", venues, placeholder="All venues")
    with f_col4:
        access_filter = st.selectbox("Access", ["All Papers", "Open Access PDF", "Abstract Only"])

    # Apply filters
    filtered = df[(df["year"] >= yr_filter[0]) & (df["year"] <= yr_filter[1])]
    if selected_venues:
        filtered = filtered[filtered["journal"].isin(selected_venues)]
    if access_filter == "Open Access PDF":
        filtered = filtered[filtered["open_access"]]
    elif access_filter == "Abstract Only":
        filtered = filtered[~filtered["open_access"]]

    if query.strip():
        q = query.strip()
        if search_mode == "Paper Name":
            filtered = filtered[filtered["title"].str.contains(q, case=False, na=False)]
        elif search_mode == "Journal / Venue":
            filtered = filtered[filtered["journal"].str.contains(q, case=False, na=False)]
        elif search_mode == "Keywords":
            filtered = filtered[
                filtered["abstract"].str.contains(q, case=False, na=False)
                | filtered["title"].str.contains(q, case=False, na=False)
            ]
        else:
            filtered = filtered[
                filtered["title"].str.contains(q, case=False, na=False)
                | filtered["journal"].str.contains(q, case=False, na=False)
                | filtered["abstract"].str.contains(q, case=False, na=False)
            ]

    filtered = filtered.sort_values(by=["open_access", "year"], ascending=[False, False])
    total_results = len(filtered)
    st.caption(f"Found {total_results:,} matching papers")

    if filtered.empty:
        st.info("No papers match your current filters.")
        return

    st.divider()

    # 2. PAGINATED LIST OF ALL MATCHING PAPERS
    page_size = 25
    total_pages = max(1, (total_results + page_size - 1) // page_size)

    if "search_page" not in st.session_state:
        st.session_state.search_page = 1
    if st.session_state.search_page > total_pages:
        st.session_state.search_page = 1

    start_idx = (st.session_state.search_page - 1) * page_size
    page_df = filtered.iloc[start_idx : start_idx + page_size]

    for idx, row in enumerate(page_df.itertuples()):
        marked_tag = (
            "<span style='background:#4285F4; color:#FFFFFF; padding:2px 8px; "
            "border-radius:999px; font-size:0.74rem; margin-left:8px;'>Marked</span>"
            if row.pmid in st.session_state.bookmarked_pmids
            else ""
        )
        title_safe = html.escape(str(row.title))
        venue_safe = html.escape(str(row.journal))

        card_html = (
            f'<div class="rag-chunk-card" style="margin-bottom:8px;">'
            f'<div style="font-size:0.98rem; font-weight:700; margin-bottom:4px;">'
            f'{title_safe}{marked_tag}</div>'
            f'<div style="font-size:0.83rem; opacity:0.82;">'
            f'<b>Venue:</b> {venue_safe} &nbsp;|&nbsp; <b>Year:</b> {row.year}</div>'
            f'</div>'
        )

        card_c1, card_c2 = st.columns([5.0, 1.3], vertical_alignment="center")
        with card_c1:
            st.markdown(card_html, unsafe_allow_html=True)
        with card_c2:
            btn_text = "Read Paper" if row.open_access else "Read Abstract"
            btn_type = "primary" if row.open_access else "secondary"
            if st.button(btn_text, key=f"open_{row.pmid}_{idx}", use_container_width=True, type=btn_type):
                st.session_state.selected_paper_pmid = row.pmid
                st.rerun()

    # Pagination controls
    if total_pages > 1:
        st.divider()
        pg1, pg2, pg3 = st.columns([1, 2, 1])
        with pg1:
            if st.button("Previous 25", use_container_width=True, disabled=st.session_state.search_page <= 1):
                st.session_state.search_page -= 1
                st.rerun()
        with pg2:
            st.markdown(
                f"<div style='text-align:center; padding-top:8px; font-weight:700;'>"
                f"Page {st.session_state.search_page} of {total_pages} ({total_results:,} papers)</div>",
                unsafe_allow_html=True,
            )
        with pg3:
            if st.button("Next 25", use_container_width=True, disabled=st.session_state.search_page >= total_pages):
                st.session_state.search_page += 1
                st.rerun()
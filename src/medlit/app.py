import os
import json
import requests
import pandas as pd
import plotly.express as px
import streamlit as st
from lxml import etree
from src.medlit.agent.graph import build_graph
from src.medlit.ui.paper_reader import render_paper_explorer

STATS_CSV = "data/processed/parkinsons_yearly_stats.csv"
ABSTRACTS_JSONL = "data/processed/parkinsons_abstracts_meta.jsonl"
GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}

BRAND_PALETTE = [
    "#4285F4", "#EA4335", "#34A853", "#FBBC05",
    "#1A73E8", "#D93025", "#188038", "#F9AB00",
    "#669DF6", "#EE675C", "#5BB974", "#FCC934",
]

st.set_page_config(
    page_title="MedLit Copilot v2.3 — Parkinson's Research Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------
# SIDEBAR: THEME, FILTERS & LIVE STATUS
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("<div class='sidebar-brand'>MedLit Copilot <span class='version-pill'>v2.3</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='sidebar-sub'>Parkinson's Full-Text RAG & Analytics</div>", unsafe_allow_html=True)
    st.divider()

    dark_mode = st.toggle("Dark Mode", value=True)
    st.divider()

    df_stats = pd.read_csv(STATS_CSV) if os.path.exists(STATS_CSV) else pd.DataFrame()

    if not df_stats.empty:
        min_yr, max_yr = int(df_stats["Year"].min()), int(df_stats["Year"].max())
        year_range = st.slider(
            "Publication Year Range",
            min_value=min_yr,
            max_value=max_yr,
            value=(2010, max_yr),
        )

        sub_topics = sorted(
            [t for t in df_stats["Topic"].unique() if t != "Overall Parkinson's"]
        )
        default_topics = [
            t for t in [
                "Alpha-Synuclein & Lewy Pathology",
                "Biomarkers & Diagnostics",
                "Neuroinflammation & Glia",
                "Myelination & White Matter",
                "Genetics (LRRK2/GBA/PINK1/PRKN)",
            ]
            if t in sub_topics
        ]
        selected_topics = st.multiselect(
            "Focus Sub-Topics",
            options=sub_topics,
            default=default_topics or sub_topics[:5],
        )
    else:
        year_range = (2000, 2026)
        selected_topics = []

    st.divider()
    st.markdown(
        """
        <div class="status-box">
            <div class="status-title">System Telemetry</div>
            <div class="status-row"><span class="pulse-dot green"></span>Qdrant: parkinsons_fulltext</div>
            <div class="status-row"><span class="pulse-dot blue"></span>GROBID: localhost:8070</div>
            <div class="status-row"><span class="pulse-dot red"></span>LangGraph: Query Rewrite Active</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ---------------------------------------------------------
# HIGH-CONTRAST THEME TOKENS (DARK & LIGHT)
# ---------------------------------------------------------
if dark_mode:
    BG_GRADIENT = "radial-gradient(circle at 15% 15%, #131B2E 0%, #090D16 55%, #060910 100%)"
    SIDEBAR_BG = "rgba(13, 18, 30, 0.94)"
    CARD_BG = "linear-gradient(145deg, rgba(28, 37, 56, 0.78), rgba(16, 22, 36, 0.88))"
    TICKER_BG = "rgba(20, 28, 46, 0.85)"
    INPUT_BG = "#141C2E"
    BORDER_COLOR = "rgba(255, 255, 255, 0.14)"
    GLOSS_HIGHLIGHT = "inset 0 1px 0 rgba(255, 255, 255, 0.18)"
    TEXT_PRIMARY = "#FFFFFF"
    TEXT_SECONDARY = "#CBD5E1"
    AXIS_COLOR = "#E2E8F0"
    PLOTLY_TEMPLATE = "plotly_dark"
    SHADOW = "0 10px 30px rgba(0, 0, 0, 0.45)"
    CHUNK_CARD_BG = "rgba(0, 0, 0, 0.32)"
    CHUNK_TEXT = "#E2E8F0"
    NAV_INACTIVE_BG = "linear-gradient(145deg, rgba(28, 37, 56, 0.82), rgba(16, 22, 36, 0.92))"
else:
    BG_GRADIENT = "radial-gradient(circle at 15% 15%, #EEF4FF 0%, #F8FAFC 55%, #EFF3F9 100%)"
    SIDEBAR_BG = "rgba(255, 255, 255, 0.95)"
    CARD_BG = "linear-gradient(145deg, rgba(255, 255, 255, 0.95), rgba(241, 245, 251, 0.88))"
    TICKER_BG = "rgba(255, 255, 255, 0.92)"
    INPUT_BG = "#FFFFFF"
    BORDER_COLOR = "rgba(15, 23, 42, 0.14)"
    GLOSS_HIGHLIGHT = "inset 0 1px 0 rgba(255, 255, 255, 0.95)"
    TEXT_PRIMARY = "#0F172A"
    TEXT_SECONDARY = "#334155"
    AXIS_COLOR = "#1E293B"
    PLOTLY_TEMPLATE = "plotly_white"
    SHADOW = "0 10px 28px rgba(15, 23, 42, 0.08)"
    CHUNK_CARD_BG = "rgba(255, 255, 255, 0.92)"
    CHUNK_TEXT = "#0F172A"
    NAV_INACTIVE_BG = "linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(238, 242, 248, 0.9))"

# ---------------------------------------------------------
# GLOSSY, ROUNDED & ANIMATED CSS
# ---------------------------------------------------------
st.markdown(
    f"""
    <style>
        @keyframes gradientShift {{
            0% {{ background-position: 0% 50%; }}
            50% {{ background-position: 100% 50%; }}
            100% {{ background-position: 0% 50%; }}
        }}
        @keyframes pulseGlow {{
            0% {{ transform: scale(0.95); opacity: 0.7; }}
            50% {{ transform: scale(1.25); opacity: 1; }}
            100% {{ transform: scale(0.95); opacity: 0.7; }}
        }}
        @keyframes fadeInUp {{
            from {{ opacity: 0; transform: translateY(10px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        @keyframes marqueeScroll {{
            0% {{ transform: translateX(0%); }}
            100% {{ transform: translateX(-50%); }}
        }}

        /* App & Sidebar Canvas */
        .stApp {{
            background: {BG_GRADIENT};
            color: {TEXT_PRIMARY};
        }}
        section[data-testid="stSidebar"] {{
            background: {SIDEBAR_BG};
            border-right: 1px solid {BORDER_COLOR};
            backdrop-filter: blur(16px);
        }}
        section[data-testid="stSidebar"] label,
        section[data-testid="stSidebar"] p,
        section[data-testid="stSidebar"] span {{
            color: {TEXT_PRIMARY} !important;
        }}

        /* Animated Hero Banner */
        .hero-banner {{
            background: linear-gradient(120deg, #4285F4, #1A73E8, #34A853, #EA4335);
            background-size: 300% 300%;
            animation: gradientShift 10s ease infinite;
            border-radius: 24px;
            padding: 26px 32px;
            margin-bottom: 16px;
            box-shadow: {SHADOW}, inset 0 1px 1px rgba(255,255,255,0.35);
            color: #FFFFFF !important;
        }}
        .hero-title {{
            font-size: 1.95rem;
            font-weight: 800;
            letter-spacing: -0.02em;
            color: #FFFFFF !important;
            margin: 0;
        }}
        .hero-sub {{
            font-size: 0.95rem;
            color: rgba(255, 255, 255, 0.92) !important;
            margin-top: 6px;
        }}

        /* Continuous Horizontal Trending Topics Ticker */
        .ticker-wrapper {{
            display: flex;
            align-items: center;
            background: {TICKER_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 999px;
            padding: 8px 16px;
            margin-bottom: 22px;
            overflow: hidden;
            box-shadow: {SHADOW}, {GLOSS_HIGHLIGHT};
            backdrop-filter: blur(12px);
        }}
        .ticker-label {{
            flex-shrink: 0;
            background: linear-gradient(135deg, #EA4335, #D93025);
            color: #FFFFFF !important;
            font-size: 0.75rem;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            padding: 6px 14px;
            border-radius: 999px;
            margin-right: 16px;
            box-shadow: 0 4px 12px rgba(234, 67, 53, 0.35);
        }}
        .ticker-viewport {{
            overflow: hidden;
            width: 100%;
            mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent);
            -webkit-mask-image: linear-gradient(to right, transparent, black 4%, black 96%, transparent);
        }}
        .ticker-track {{
            display: flex;
            width: max-content;
            gap: 18px;
            animation: marqueeScroll 32s linear infinite;
        }}
        .ticker-viewport:hover .ticker-track {{
            animation-play-state: paused;
        }}
        .ticker-pill {{
            display: inline-flex;
            align-items: center;
            gap: 8px;
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 999px;
            padding: 5px 14px;
            font-size: 0.84rem;
            font-weight: 600;
            color: {TEXT_PRIMARY} !important;
            white-space: nowrap;
        }}
        .ticker-growth {{
            color: #34A853 !important;
            font-weight: 800;
            font-size: 0.8rem;
        }}

        /* Big Horizontal Rounded Navigation Buttons */
        div[data-testid="stButton"] > button {{
            border-radius: 999px !important;
            font-weight: 700 !important;
            transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease !important;
        }}
        div[data-testid="stButton"] > button[kind="primary"] {{
            background: linear-gradient(135deg, #4285F4 0%, #1A73E8 100%) !important;
            color: #FFFFFF !important;
            border: 1px solid rgba(255, 255, 255, 0.28) !important;
            min-height: 56px !important;
            font-size: 1.02rem !important;
            letter-spacing: 0.01em !important;
            box-shadow: 0 8px 22px rgba(66, 133, 244, 0.42), inset 0 1px 1px rgba(255,255,255,0.45) !important;
        }}
        div[data-testid="stButton"] > button[kind="secondary"] {{
            background: {NAV_INACTIVE_BG} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_COLOR} !important;
            min-height: 56px !important;
            font-size: 1.02rem !important;
            box-shadow: {SHADOW}, {GLOSS_HIGHLIGHT} !important;
        }}
        div[data-testid="stButton"] > button:hover {{
            transform: translateY(-3px) !important;
            border-color: #4285F4 !important;
            box-shadow: 0 12px 26px rgba(66, 133, 244, 0.35), inset 0 1px 1px rgba(255,255,255,0.5) !important;
        }}

        /* Sidebar Branding & Animated Status Box */
        .sidebar-brand {{
            font-size: 1.3rem;
            font-weight: 800;
            color: {TEXT_PRIMARY};
        }}
        .version-pill {{
            background: linear-gradient(135deg, #4285F4, #1A73E8);
            color: #FFFFFF !important;
            font-size: 0.72rem;
            padding: 3px 10px;
            border-radius: 999px;
            margin-left: 6px;
            vertical-align: middle;
        }}
        .sidebar-sub {{
            font-size: 0.82rem;
            color: {TEXT_SECONDARY};
            margin-top: 4px;
        }}
        .status-box {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 18px;
            padding: 14px 16px;
            box-shadow: {GLOSS_HIGHLIGHT};
        }}
        .status-title {{
            font-size: 0.82rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: {TEXT_SECONDARY};
            margin-bottom: 10px;
        }}
        .status-row {{
            font-size: 0.83rem;
            color: {TEXT_PRIMARY};
            margin-bottom: 6px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .pulse-dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
            animation: pulseGlow 2.2s infinite ease-in-out;
        }}
        .pulse-dot.green {{ background: #34A853; box-shadow: 0 0 8px #34A853; }}
        .pulse-dot.blue {{ background: #4285F4; box-shadow: 0 0 8px #4285F4; }}
        .pulse-dot.red {{ background: #EA4335; box-shadow: 0 0 8px #EA4335; }}

        /* Glossy Rounded KPI Metric Cards */
        div[data-testid="stMetric"] {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 20px;
            padding: 18px 22px;
            box-shadow: {SHADOW}, {GLOSS_HIGHLIGHT};
            backdrop-filter: blur(12px);
            transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
            animation: fadeInUp 0.5s ease both;
        }}
        div[data-testid="stMetric"]:hover {{
            transform: translateY(-4px);
            border-color: #4285F4;
        }}
        div[data-testid="stMetricLabel"] p {{
            color: {TEXT_SECONDARY} !important;
            font-size: 0.86rem !important;
            font-weight: 700 !important;
        }}
        div[data-testid="stMetricValue"] {{
            color: {TEXT_PRIMARY} !important;
            font-size: 1.75rem !important;
            font-weight: 800 !important;
        }}

        /* Rounded Inputs, Chat & Expanders */
        .stTextInput input, .stChatInput textarea {{
            background-color: {INPUT_BG} !important;
            color: {TEXT_PRIMARY} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 16px !important;
        }}
        div[data-testid="stExpander"] {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 18px;
            box-shadow: {GLOSS_HIGHLIGHT};
        }}
        div[data-testid="stChatMessage"] {{
            background: {CARD_BG};
            border: 1px solid {BORDER_COLOR};
            border-radius: 18px;
            padding: 14px 18px;
            box-shadow: {GLOSS_HIGHLIGHT};
            animation: fadeInUp 0.35s ease both;
        }}

        /* Dual-Color RAG Display: Exact Evidence (Emerald) vs RAG Summary (Indigo) */
        .rag-exact-container {{
            background: rgba(16, 185, 129, 0.08) !important;
            border: 1.5px solid rgba(16, 185, 129, 0.35) !important;
            border-radius: 18px;
            padding: 18px 22px;
            margin-bottom: 18px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
            animation: fadeInUp 0.4s ease both;
        }}
        .rag-summary-container {{
            background: rgba(99, 102, 241, 0.08) !important;
            border: 1.5px solid rgba(99, 102, 241, 0.35) !important;
            border-radius: 18px;
            padding: 18px 22px;
            margin-top: 14px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.12);
            animation: fadeInUp 0.5s ease both;
        }}
        .rag-chunk-card {{
            background: {CHUNK_CARD_BG} !important;
            border: 1px solid {BORDER_COLOR} !important;
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 12px;
            transition: transform 0.2s ease, border-color 0.2s ease;
        }}
        .rag-chunk-card:hover {{
            transform: translateY(-2px);
            border-color: rgba(66, 133, 244, 0.5) !important;
        }}
        .rag-chunk-card div {{
            color: {CHUNK_TEXT} !important;
        }}
        .paper-citation-badge {{
            transition: transform 0.18s ease, filter 0.18s ease;
            box-shadow: 0 1px 3px rgba(0, 0, 0, 0.12);
            white-space: normal;
            word-break: break-word;
        }}
        .paper-citation-badge:hover {{
            transform: translateY(-1px);
            filter: brightness(1.15);
        }}

        /* Enforce Crisp Text Globally */
        h1, h2, h3, h4, p, li, label {{
            color: {TEXT_PRIMARY};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)


def style_figure(fig, title: str, height: int = 440, legend_bottom: bool = True):
    """Apply crisp, high-contrast, grid-free styling with a compact bottom legend."""
    legend_cfg = (
        dict(
            orientation="h",
            yanchor="top",
            y=-0.18,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_PRIMARY, size=11),
            title_text="",
        )
        if legend_bottom
        else dict(
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TEXT_PRIMARY, size=11),
            title_font=dict(color=TEXT_PRIMARY, size=12),
        )
    )

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        title=dict(text=title, font=dict(size=17, color=TEXT_PRIMARY)),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, size=12),
        margin=dict(l=24, r=24, t=56, b=90 if legend_bottom else 30),
        hovermode="closest",
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=AXIS_COLOR, size=12),
            title_font=dict(color=TEXT_PRIMARY, size=13),
        ),
        yaxis=dict(
            showgrid=False,
            zeroline=False,
            tickfont=dict(color=AXIS_COLOR, size=12),
            title_font=dict(color=TEXT_PRIMARY, size=13),
        ),
        legend=legend_cfg,
    )
    return fig


def build_trending_ticker_html(df: pd.DataFrame) -> str:
    """Compute top trending Parkinson's topics and return a looping marquee HTML block."""
    items = []
    if not df.empty:
        sub = df[df["Topic"] != "Overall Parkinson's"]
        ranked = []
        for topic, grp in sub.groupby("Topic"):
            grp_s = grp.sort_values("Year")
            recent_avg = grp_s.tail(3)["Paper_Count"].mean()
            early_avg = max(grp_s.head(3)["Paper_Count"].mean(), 1)
            growth = round(((recent_avg - early_avg) / early_avg) * 100)
            total = int(grp_s["Paper_Count"].sum())
            ranked.append((topic, growth, total))
        ranked.sort(key=lambda x: x[1], reverse=True)
        for topic, growth, total in ranked[:8]:
            sign = "+" if growth >= 0 else ""
            items.append(
                f"<span class='ticker-pill'>{topic} "
                f"<span class='ticker-growth'>{sign}{growth}% ({total:,})</span></span>"
            )
    else:
        for topic in [
            "Alpha-Synuclein & Lewy Pathology",
            "Biomarkers & Diagnostics",
            "Neuroinflammation & Glia",
            "Myelination & White Matter",
            "Genetics (LRRK2/GBA/PINK1/PRKN)",
            "AI, Machine Learning & Neuroimaging",
        ]:
            items.append(f"<span class='ticker-pill'>{topic}</span>")

    track_html = "".join(items) * 2  # Duplicate for seamless infinite loop
    return f"""
    <div class="ticker-wrapper">
        <div class="ticker-label">Trending in PD</div>
        <div class="ticker-viewport">
            <div class="ticker-track">{track_html}</div>
        </div>
    </div>
    """


# ---------------------------------------------------------
# ANIMATED HERO HEADER, TRENDING TICKER & BIG NAV BUTTONS
# ---------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <div class="hero-title">Parkinson's Disease Research Intelligence</div>
        <div class="hero-sub">Longitudinal PubMed Analytics  |  Full-Text PMC Hybrid RAG  |  Interactive Paper Reader & Annotator</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(build_trending_ticker_html(df_stats), unsafe_allow_html=True)

VALID_SECTIONS = {"Analytics Dashboard", "Full-Text RAG Chat", "Search & Read Papers"}
if st.session_state.get("active_section") not in VALID_SECTIONS:
    st.session_state.active_section = "Analytics Dashboard"

nav_col1, nav_col2, nav_col3 = st.columns(3, gap="medium")
with nav_col1:
    if st.button(
        "Analytics Dashboard",
        use_container_width=True,
        type="primary" if st.session_state.active_section == "Analytics Dashboard" else "secondary",
    ):
        st.session_state.active_section = "Analytics Dashboard"
        st.rerun()
with nav_col2:
    if st.button(
        "Full-Text RAG Chat",
        use_container_width=True,
        type="primary" if st.session_state.active_section == "Full-Text RAG Chat" else "secondary",
    ):
        st.session_state.active_section = "Full-Text RAG Chat"
        st.rerun()
with nav_col3:
    if st.button(
        "Search & Read Papers",
        use_container_width=True,
        type="primary" if st.session_state.active_section == "Search & Read Papers" else "secondary",
    ):
        st.session_state.active_section = "Search & Read Papers"
        st.rerun()

st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 1: ANALYTICS DASHBOARD
# ---------------------------------------------------------
if st.session_state.active_section == "Analytics Dashboard":
    if df_stats.empty:
        st.warning("Run `python -m src.medlit.ingest.parkinsons_stats` to generate analytics data.")
    else:
        df_range = df_stats[
            (df_stats["Year"] >= year_range[0]) & (df_stats["Year"] <= year_range[1])
        ]
        overall_df = df_range[df_range["Topic"] == "Overall Parkinson's"]
        sub_df = df_range[df_range["Topic"].isin(selected_topics)]

        total_pd = int(overall_df["Paper_Count"].sum()) if not overall_df.empty else 0
        peak_row = overall_df.loc[overall_df["Paper_Count"].idxmax()] if not overall_df.empty else None
        selected_vol = int(sub_df["Paper_Count"].sum()) if not sub_df.empty else 0

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Total PD Papers ({year_range[0]}-{year_range[1]})", f"{total_pd:,}")
        k2.metric(
            "Peak Publication Year",
            f"{int(peak_row['Year'])}" if peak_row is not None else "N/A",
            delta=f"{int(peak_row['Paper_Count']):,} papers" if peak_row is not None else None,
        )
        k3.metric("Selected Sub-Topics Volume", f"{selected_vol:,}")
        k4.metric("Active Sub-Topics", f"{len(selected_topics)} / {df_stats['Topic'].nunique() - 1}")

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)

        if sub_df.empty:
            st.info("Select at least one sub-topic in the sidebar to render charts.")
        else:
            # 1. FULL-WIDTH ANNUAL PUBLICATION VELOCITY (BIG PLOT, COMPACT BOTTOM LEGEND)
            fig_velocity = px.area(
                sub_df,
                x="Year",
                y="Paper_Count",
                color="Topic",
                markers=True,
                color_discrete_sequence=BRAND_PALETTE,
                labels={"Paper_Count": "Publications", "Year": "Publication Year"},
            )
            fig_velocity.update_traces(line=dict(width=2.8))
            st.plotly_chart(
                style_figure(
                    fig_velocity,
                    "Annual Publication Velocity by Sub-Topic",
                    height=520,
                    legend_bottom=True,
                ),
                use_container_width=True,
            )

            # 2. HISTOGRAM ROW: YEAR-WISE TOTAL PD & TOPIC-WISE YEAR-WISE HISTOGRAM
            h1, h2 = st.columns(2, gap="large")

            with h1:
                fig_hist_total = px.histogram(
                    overall_df,
                    x="Year",
                    y="Paper_Count",
                    histfunc="sum",
                    nbins=max(overall_df["Year"].nunique(), 1),
                    color_discrete_sequence=["#4285F4"],
                    labels={"Paper_Count": "Total Papers", "Year": "Year"},
                )
                fig_hist_total.update_traces(marker_line_color="rgba(255,255,255,0.6)", marker_line_width=1)
                fig_hist_total.update_layout(bargap=0.12)
                st.plotly_chart(
                    style_figure(
                        fig_hist_total,
                        "Year-Wise Total Parkinson's Publications (Histogram)",
                        height=440,
                        legend_bottom=False,
                    ),
                    use_container_width=True,
                )

            with h2:
                hist_mode = st.radio(
                    "Topic Histogram Mode",
                    ["Stacked", "Grouped"],
                    horizontal=True,
                    label_visibility="collapsed",
                )
                fig_hist_topic = px.histogram(
                    sub_df,
                    x="Year",
                    y="Paper_Count",
                    color="Topic",
                    histfunc="sum",
                    barmode="stack" if hist_mode == "Stacked" else "group",
                    nbins=max(sub_df["Year"].nunique(), 1),
                    color_discrete_sequence=BRAND_PALETTE,
                    labels={"Paper_Count": "Papers", "Year": "Year"},
                )
                fig_hist_topic.update_layout(bargap=0.14)
                st.plotly_chart(
                    style_figure(
                        fig_hist_topic,
                        "Topic-Wise Year-Wise Publication Histogram",
                        height=405,
                        legend_bottom=True,
                    ),
                    use_container_width=True,
                )

            # 3. SHARE DONUT + MOMENTUM BUBBLE + VOLUME RANKING
            pie_df = (
                sub_df.groupby("Topic", as_index=False)["Paper_Count"]
                .sum()
                .sort_values("Paper_Count", ascending=False)
            )

            c1, c2 = st.columns([1, 1.3], gap="large")
            with c1:
                fig_pie = px.pie(
                    pie_df,
                    names="Topic",
                    values="Paper_Count",
                    hole=0.52,
                    color_discrete_sequence=BRAND_PALETTE,
                )
                fig_pie.update_traces(
                    textposition="inside",
                    textinfo="percent",
                    textfont=dict(color="#FFFFFF", size=12),
                )
                st.plotly_chart(
                    style_figure(fig_pie, "Research Share by Sub-Topic", height=440, legend_bottom=True),
                    use_container_width=True,
                )

            with c2:
                bubble_rows = []
                for topic, grp in sub_df.groupby("Topic"):
                    grp_sorted = grp.sort_values("Year")
                    total = int(grp_sorted["Paper_Count"].sum())
                    avg_yr = round(grp_sorted["Paper_Count"].mean(), 1)
                    first_val = max(int(grp_sorted["Paper_Count"].iloc[0]), 1)
                    last_val = int(grp_sorted["Paper_Count"].iloc[-1])
                    growth_pct = round(((last_val - first_val) / first_val) * 100, 1)
                    bubble_rows.append({
                        "Topic": topic,
                        "Total_Papers": total,
                        "Avg_Per_Year": avg_yr,
                        "Growth_Pct": growth_pct,
                    })
                df_bubble = pd.DataFrame(bubble_rows)

                fig_bubble = px.scatter(
                    df_bubble,
                    x="Growth_Pct",
                    y="Avg_Per_Year",
                    size="Total_Papers",
                    color="Topic",
                    hover_name="Topic",
                    size_max=55,
                    color_discrete_sequence=BRAND_PALETTE,
                    labels={
                        "Growth_Pct": f"Growth Rate ({year_range[0]} to {year_range[1]}, %)",
                        "Avg_Per_Year": "Average Papers / Year",
                    },
                )
                st.plotly_chart(
                    style_figure(fig_bubble, "Sub-Topic Momentum Bubble Chart", height=440, legend_bottom=True),
                    use_container_width=True,
                )

            # 4. FULL-WIDTH CUMULATIVE VOLUME RANKING
            bar_df = pie_df.sort_values("Paper_Count", ascending=True)
            fig_bar = px.bar(
                bar_df,
                x="Paper_Count",
                y="Topic",
                orientation="h",
                text_auto=".2s",
                color="Topic",
                color_discrete_sequence=BRAND_PALETTE,
                labels={"Paper_Count": "Total Publications", "Topic": ""},
            )
            fig_bar.update_traces(textfont=dict(color=TEXT_PRIMARY, size=12))
            fig_bar.update_layout(showlegend=False)
            st.plotly_chart(
                style_figure(fig_bar, "Cumulative Volume Ranking", height=380, legend_bottom=False),
                use_container_width=True,
            )

        st.divider()
        st.subheader("Searchable Parkinson's Abstracts Explorer")
        if os.path.exists(ABSTRACTS_JSONL):
            with open(ABSTRACTS_JSONL, "r", encoding="utf-8") as f:
                abs_rows = [json.loads(line) for line in f]
            df_abs = pd.DataFrame(abs_rows)

            kw = st.text_input(
                "Filter abstracts by keyword (e.g., oligodendrocyte, myelin, alpha-synuclein, LRRK2):"
            )
            if kw:
                mask = (
                    df_abs["title"].str.contains(kw, case=False, na=False)
                    | df_abs["abstract"].str.contains(kw, case=False, na=False)
                )
                df_abs = df_abs[mask]

            st.caption(f"Showing {len(df_abs):,} matching papers")
            st.dataframe(
                df_abs[["pmid", "year", "journal", "title", "abstract"]],
                use_container_width=True,
                height=360,
            )


@st.cache_resource
def get_cached_agent():
    """Caches the compiled LangGraph agent across Streamlit reruns to prevent duplicate connections."""
    return build_graph()


def render_sources(sources: list, msg_idx: int):
    """Show all sources if <= 5; if > 5, show first 5 with a 'See more' toggle."""
    if not sources:
        return

    total = len(sources)
    state_key = f"expand_sources_{msg_idx}"
    if state_key not in st.session_state:
        st.session_state[state_key] = False

    visible_sources = sources if (total <= 5 or st.session_state[state_key]) else sources[:5]

    st.markdown(
        f"<div style='font-size:0.9rem; font-weight:700; margin:14px 0 8px 0; color:#34A853;'>"
        f"Original Sources ({len(visible_sources)} of {total} shown)</div>",
        unsafe_allow_html=True,
    )

    for idx, src in enumerate(visible_sources, 1):
        title = src.get("title", "Untitled")
        venue = src.get("journal") or src.get("venue") or "Journal N/A"
        pmid = src.get("pmid", "N/A")
        year = src.get("year", "")
        section = src.get("section", "Body")
        raw_text = src.get("text", "").strip()
        if raw_text.startswith("[") and "]\n" in raw_text:
            raw_text = raw_text.split("]\n", 1)[1]

        st.markdown(
            f"""
            <div class="rag-chunk-card">
                <div style="font-size:0.86rem; font-weight:700; margin-bottom:6px;">
                    <span class="paper-citation-badge" style="background:#4285F4; color:#FFFFFF; padding:2px 8px; border-radius:999px; margin-right:6px;">[{idx}]</span>
                    {title}
                    <span style="opacity:0.75; font-weight:500;"> ({venue} | {year} | PMID: {pmid} | {section})</span>
                </div>
                <div style="font-size:0.84rem; line-height:1.5; opacity:0.92;">
                    {raw_text[:450]}{"..." if len(raw_text) > 450 else ""}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if total > 5:
        btn_label = (
            "Show fewer sources"
            if st.session_state[state_key]
            else f"See more sources (+{total - 5})"
        )
        if st.button(btn_label, key=f"btn_{state_key}", type="secondary"):
            st.session_state[state_key] = not st.session_state[state_key]
            st.rerun()


# ---------------------------------------------------------
# SECTION 2: FULL-TEXT RAG CHAT (WITH MEMORY)
# ---------------------------------------------------------
if st.session_state.active_section == "Full-Text RAG Chat":
    if "agent" not in st.session_state:
        with st.spinner("Loading LangGraph agent and Qdrant vector store..."):
            st.session_state.agent = get_cached_agent()

    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Ask a question about Parkinson's disease (oligodendrocytes, myelination, alpha-synuclein, biomarkers). Answers are grounded in indexed full-text literature with numbered source citations.",
                "sources": [],
            }
        ]

    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                render_sources(msg["sources"], msg_idx=i)

    if user_query := st.chat_input(
        "Example: How does alpha-synuclein pathology affect oligodendrocytes and myelination?"
    ):
        st.session_state.messages.append({"role": "user", "content": user_query, "sources": []})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Rewriting query, retrieving full-text sections, and generating..."):
                history_payload = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]
                final_state = st.session_state.agent.invoke({
                    "query": user_query,
                    "chat_history": history_payload,
                })
                answer = final_state["answer"]
                sources = final_state.get("final_chunks", [])

                st.markdown(answer)
                render_sources(sources, msg_idx=len(st.session_state.messages))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
# ---------------------------------------------------------
# SECTION 3: SEARCH & READ PAPERS
# ---------------------------------------------------------
if st.session_state.active_section == "Search & Read Papers":
    render_paper_explorer()

#------------------------------------------------------    


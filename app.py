# app.py
# Single-page Streamlit frontend for Noesis AI.

import streamlit as st

from graph import stream_research
from tools import get_chroma_collection, get_ingested_sources
import config

st.set_page_config(page_title="Noesis AI", page_icon="🧠", layout="centered")

_THEME_CSS = """
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&family=Source+Serif+4:ital,wght@0,400;0,500;1,400&display=swap" rel="stylesheet">
<style>
:root{
    --paper: #EFECE2;
    --paper-2: #E4E0D2;
    --ink: #171F26;
    --ink-soft: #4A5560;
    --blue: #2C56C9;
    --line: rgba(23,31,38,0.22);
    --stamp: #3C6E4F;
    --warn: #B45309;
    --danger: #B42318;
    --mono: 'JetBrains Mono', monospace;
    --serif: 'Source Serif 4', serif;
}

/* page background: graph paper grid, same as the portfolio */
.stApp{
    background:
      linear-gradient(rgba(23,31,38,0.08) 1px, transparent 1px) 0 0 / 40px 40px,
      linear-gradient(90deg, rgba(23,31,38,0.08) 1px, transparent 1px) 0 0 / 40px 40px,
      var(--paper);
}

html, body, [class*="css"]{ font-family: var(--serif); color: var(--ink); }

/* ---- headings / eyebrow labels ---- */
.eyebrow{
    font-family: var(--mono); font-size: 12.5px; letter-spacing: 0.14em;
    color: var(--blue); text-transform: uppercase; font-weight: 600;
    display:flex; align-items:center; gap:10px; margin-bottom: 6px;
}
.eyebrow::before{ content:''; width: 26px; height:1px; background: var(--blue); }

h1.headline{
    font-family: var(--mono); font-weight: 800;
    font-size: clamp(30px, 5vw, 46px);
    line-height: 1.05; letter-spacing: -0.01em;
    margin: 0 0 10px; color: var(--ink);
}
h1.headline .u{ color: var(--blue); }

.sub{ font-family: var(--serif); font-size: 17px; color: var(--ink-soft); max-width: 600px; margin: 0 0 28px; }

/* ---- section divider, same "dimension line" motif ---- */
.dim{ display:flex; align-items:center; gap: 14px; margin: 32px 0 20px; }
.dim .line{ flex:1; height:1px; background: var(--line); }
.dim .label{ font-family: var(--mono); font-size: 11px; letter-spacing: 0.12em; color: var(--ink-soft); white-space: nowrap; text-transform: uppercase; }

/* ---- bordered containers (st.container(border=True)) restyled as spec sheets ---- */
div[data-testid="stVerticalBlockBorderWrapper"]{
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
    background: rgba(255,255,255,0.35) !important;
}

/* ---- text input: underline style, not a big white pill ---- */
div[data-testid="stTextInput"] input{
    font-family: var(--mono); font-size: 15px;
    background: transparent;
    border: none; border-bottom: 1.5px solid var(--ink);
    border-radius: 0; padding: 8px 2px; color: var(--ink);
}
div[data-testid="stTextInput"] input:focus{
    border-bottom: 1.5px solid var(--blue);
    box-shadow: none;
}
div[data-testid="stTextInput"] label{
    font-family: var(--mono) !important; font-size: 11.5px !important;
    letter-spacing: 0.08em; text-transform: uppercase; color: var(--ink-soft) !important;
}

/* ---- buttons: solid ink rectangle, matches .btn.solid ---- */
div.stButton > button, div.stDownloadButton > button{
    font-family: var(--mono); font-size: 13.5px; font-weight: 600;
    background: var(--ink); color: var(--paper);
    border: 1px solid var(--ink); border-radius: 3px;
    padding: 10px 22px; letter-spacing: 0.02em;
    transition: transform .15s, background .15s, border-color .15s;
}
div.stButton > button:hover, div.stDownloadButton > button:hover{
    background: var(--blue); border-color: var(--blue); color: var(--paper);
    transform: translateY(-2px);
}

/* ---- spec sheet rows (sidebar settings + sources) ---- */
.spec-row{
    display:flex; justify-content:space-between; gap: 12px;
    padding: 9px 0; border-bottom: 1px dashed var(--line);
    font-family: var(--mono); font-size: 12.5px;
}
.spec-row:last-child{ border-bottom:none; }
.spec-row .k{ color: var(--ink-soft); letter-spacing: 0.04em; text-transform: uppercase; font-size:11px; padding-top:2px; }
.spec-row .v{ text-align:right; font-weight: 500; }

.tag{
    font-family: var(--mono); font-size: 11px; padding: 4px 9px;
    border: 1px dashed var(--line); border-radius: 3px; color: var(--ink-soft);
    display:inline-block; margin: 2px 4px 2px 0;
}

.badge-verified{ color: var(--stamp); font-weight: 700; }
.badge-unverified{ color: var(--warn); font-weight: 700; }
.badge-contradicted{ color: var(--danger); font-weight: 700; }

/* sidebar */
section[data-testid="stSidebar"]{
    background: var(--paper-2);
    border-right: 1px solid var(--line);
}
section[data-testid="stSidebar"] .stMarkdown{ font-family: var(--serif); }

/* status widget (live pipeline steps) - restyle to match spec-sheet look */
div[data-testid="stStatusWidget"], div[data-testid="stExpander"]{
    border: 1px solid var(--line) !important;
    border-radius: 4px !important;
    background: rgba(255,255,255,0.35) !important;
}
</style>
"""

# Strip blank lines so Streamlit doesn't cut the raw-HTML block short.
_THEME_CSS = "\n".join(line for line in _THEME_CSS.splitlines() if line.strip() != "")
st.markdown(_THEME_CSS, unsafe_allow_html=True)

# Header
st.markdown('<div class="eyebrow">Self-Critiquing Research Agent</div>', unsafe_allow_html=True)
st.markdown('<h1 class="headline">Noesis <span class="u">AI</span></h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">Grounds itself in your documents first, fills gaps with live web '
    'search, drafts a report, critiques its own work, rewrites if needed, then '
    'fact-checks its own claims before handing it to you.</p>',
    unsafe_allow_html=True,
)

# Sidebar: knowledge base status + settings, shown as spec sheets
with st.sidebar:
    st.markdown('<div class="eyebrow">Knowledge Base</div>', unsafe_allow_html=True)
    try:
        count = get_chroma_collection().count()
        sources = get_ingested_sources()
    except Exception:
        count, sources = 0, []

    st.markdown(f"""
    <div class="spec-row"><span class="k">Chunks stored</span><span class="v">{count}</span></div>
    <div class="spec-row"><span class="k">Status</span><span class="v" style="color:{'var(--stamp)' if count else 'var(--warn)'}">{'Active' if count else 'Empty'}</span></div>
    """, unsafe_allow_html=True)

    if sources:
        st.markdown("<div style='margin-top:10px;'>", unsafe_allow_html=True)
        for s in sources:
            st.markdown(f'<span class="tag">📄 {s}</span>', unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.caption("No documents ingested yet. Add PDFs/TXT to `documents/` "
                   "and run `python ingest.py`, then refresh.")

    st.markdown('<div class="dim"><div class="line"></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow">Settings In Use</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="spec-row"><span class="k">Pass score</span><span class="v">{config.CRITIQUE_PASS_SCORE}/10</span></div>
    <div class="spec-row"><span class="k">Max refine attempts</span><span class="v">{config.MAX_REFINE_ITERATIONS}</span></div>
    <div class="spec-row"><span class="k">Claims verified</span><span class="v">{config.MAX_CLAIMS_TO_VERIFY}</span></div>
    <div class="spec-row"><span class="k">Local match threshold</span><span class="v">{config.DISTANCE_THRESHOLD}</span></div>
    """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Main: topic input (real container, properly wraps the widget)
# ------------------------------------------------------------------
with st.container(border=True):
    topic = st.text_input("Research topic", placeholder="e.g. Impact of quick commerce apps on local retail")
    use_local = st.checkbox("Use local knowledge base", value=True,
                             help="Turn off to force web-only research, ignoring any uploaded documents.")
    run_clicked = st.button("🔍  Start Research")

# Run pipeline with LIVE streaming progress
if run_clicked:
    if not topic.strip():
        st.warning("Please enter a topic first.")
    else:
        st.markdown('<div class="dim"><div class="line"></div><div class="label">Pipeline</div><div class="line"></div></div>', unsafe_allow_html=True)

        final_state = None
        seen_log_lines = 0

        with st.status("Starting research pipeline...", expanded=True) as status:
            for node_name, state in stream_research(topic, use_local):
                # Print only the NEW log lines since the last update
                new_lines = state["log"][seen_log_lines:]
                for line in new_lines:
                    st.write(f"▸ {line}")
                seen_log_lines = len(state["log"])

                # Update the status label based on which node just ran
                labels = {
                    "gather_context": "Gathering context (local knowledge + web)...",
                    "write_draft": "Writing draft report...",
                    "critique_draft": f"Critic scored draft: {state['critique_score']}/10",
                    "refine_draft": "Rewriting report based on feedback...",
                    "verify_claims": "Verifying claims against evidence...",
                }
                status.update(label=labels.get(node_name, "Working..."))
                final_state = state

            status.update(label="Research complete.", state="complete", expanded=False)

        st.markdown('<div class="dim"><div class="line"></div><div class="label">Result</div><div class="line"></div></div>', unsafe_allow_html=True)

        with st.container(border=True):
            st.markdown('<div class="eyebrow">Research Report</div>', unsafe_allow_html=True)
            st.markdown(
                f"Final critic score: **{final_state['critique_score']}/10** "
                f"&nbsp;·&nbsp; refine attempts used: **{final_state['iteration']}**"
            )
            st.markdown("---")
            st.markdown(final_state["final_report"])

        st.download_button(
            "⬇️  Download report as Markdown",
            data=final_state["final_report"],
            file_name=f"{topic[:40].strip().replace(' ', '_')}_report.md",
            mime="text/markdown",
        )

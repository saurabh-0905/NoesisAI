# graph.py
# ------------------------------------------------------------------
# This is the brain of Noesis AI - the actual research pipeline.
#
# Flow (5 steps):
#   1. gather_context   -> check our own documents first, use web search
#                          only to fill in the gaps
#   2. write_draft       -> write a structured report from that context
#   3. critique_draft    -> an LLM grades its own report out of 10
#   4. refine_draft       -> if the grade is too low, rewrite the report
#                          using the critic's feedback (loops back to
#                          critique, up to a max number of times)
#   5. verify_claims      -> pull out the key factual claims and check
#                          each one against real sources
#
# Everything is stored in one shared "state" dictionary that gets
# passed from node to node.
# ------------------------------------------------------------------

import os
from typing import TypedDict, List, Dict
from langgraph.graph import StateGraph, END
from langchain_groq import ChatGroq
from dotenv import load_dotenv

import config
from tools import query_local_knowledge, web_search

load_dotenv()

llm = ChatGroq(
    model=config.GROQ_MODEL,
    temperature=config.LLM_TEMPERATURE,
    api_key=os.getenv("GROQ_API_KEY"),
)


# ------------------------------------------------------------------
# Shared state - every node reads from / writes to this
# ------------------------------------------------------------------
class ResearchState(TypedDict):
    topic: str
    use_local: bool           # NEW: whether to check ChromaDB at all this run
    context: str              # combined local + web text handed to the writer
    sources: List[Dict]       # list of {label, source} for the final report
    used_web: bool

    draft: str
    critique_score: float
    critique_feedback: str
    iteration: int

    claims: List[str]
    verdicts: List[Dict]      # [{claim, verdict, note}]

    final_report: str
    log: List[str]            # human-readable trail shown in the UI


# ------------------------------------------------------------------
# Node 1: gather_context
# ------------------------------------------------------------------
def gather_context(state: ResearchState) -> ResearchState:
    topic = state["topic"]
    log = state.get("log", [])

    local_chunks = []
    if state.get("use_local", True):
        local_chunks = query_local_knowledge(topic)
    else:
        log.append("Local knowledge base skipped (web-only mode selected).")

    sources = []
    context_parts = []
    used_web = False

    if local_chunks:
        log.append(f"Found {len(local_chunks)} relevant chunk(s) in local knowledge base.")
        for c in local_chunks:
            context_parts.append(c["text"])
            sources.append({"label": "local document", "source": c["source"]})
    elif state.get("use_local", True):
        log.append("No relevant local knowledge found.")

    # Fall back to (or supplement with) the web if local knowledge is
    # missing or looks thin (fewer than 2 chunks).
    if len(local_chunks) < 2:
        log.append("Searching the web for more information...")
        web_results = web_search(topic)
        used_web = True
        for r in web_results:
            context_parts.append(f"{r['title']}: {r['content']}")
            sources.append({"label": "web", "source": r["url"]})
        log.append(f"Found {len(web_results)} web result(s).")

    return {
        **state,
        "context": "\n\n---\n\n".join(context_parts),
        "sources": sources,
        "used_web": used_web,
        "log": log,
    }


# ------------------------------------------------------------------
# Node 2: write_draft
# ------------------------------------------------------------------
def write_draft(state: ResearchState) -> ResearchState:
    log = state.get("log", [])
    log.append("Writing draft report...")

    prompt = f"""You are a research report writer.
Topic: {state['topic']}

Use ONLY the information below. Write a clear, well-structured
Markdown report with these sections: Introduction, Key Findings,
Conclusion.

Information:
{state['context']}
"""
    response = llm.invoke(prompt)
    return {**state, "draft": response.content, "log": log}


# ------------------------------------------------------------------
# Node 3: critique_draft
# ------------------------------------------------------------------
def critique_draft(state: ResearchState) -> ResearchState:
    log = state.get("log", [])
    log.append(f"Critiquing draft (attempt {state['iteration'] + 1})...")

    prompt = f"""You are a strict editor reviewing a research report.

Report:
{state['draft']}

Score it from 0 to 10 on accuracy, completeness, and clarity.
Respond in EXACTLY this format:
SCORE: <number>
FEEDBACK: <one short paragraph of specific, actionable feedback>
"""
    response = llm.invoke(prompt).content

    score = 5.0  # safe default if parsing fails
    feedback = response
    for line in response.splitlines():
        if line.upper().startswith("SCORE"):
            try:
                score = float(line.split(":", 1)[1].strip())
            except (ValueError, IndexError):
                pass
        if line.upper().startswith("FEEDBACK"):
            feedback = line.split(":", 1)[1].strip()

    log.append(f"Critic score: {score}/10")

    return {
        **state,
        "critique_score": score,
        "critique_feedback": feedback,
        "log": log,
    }


# ------------------------------------------------------------------
# Node 4: refine_draft
# (only runs if the critique score was too low)
# ------------------------------------------------------------------
def refine_draft(state: ResearchState) -> ResearchState:
    log = state.get("log", [])
    log.append("Score too low - rewriting report based on feedback...")

    prompt = f"""Rewrite the report below to fix the issues raised in the feedback.
Keep using only the same source information - do not invent new facts.

Original report:
{state['draft']}

Editor feedback to address:
{state['critique_feedback']}

Source information (for reference, do not exceed it):
{state['context']}
"""
    response = llm.invoke(prompt)
    return {
        **state,
        "draft": response.content,
        "iteration": state["iteration"] + 1,
        "log": log,
    }


def should_refine(state: ResearchState) -> str:
    """Decides whether to loop back and rewrite, or move on to verification."""
    good_enough = state["critique_score"] >= config.CRITIQUE_PASS_SCORE
    out_of_tries = state["iteration"] >= config.MAX_REFINE_ITERATIONS
    return "verify" if (good_enough or out_of_tries) else "refine"


# ------------------------------------------------------------------
# Node 5: verify_claims
# (extraction + verification combined into ONE node, on purpose -
# keeps the pipeline simple instead of splitting into two agents)
# ------------------------------------------------------------------
def verify_claims(state: ResearchState) -> ResearchState:
    log = state.get("log", [])
    log.append("Extracting key claims to fact-check...")

    extract_prompt = f"""Read this report and list the {config.MAX_CLAIMS_TO_VERIFY} most
important factual claims it makes. One claim per line, no numbering, no extra text.

Report:
{state['draft']}
"""
    claims_raw = llm.invoke(extract_prompt).content
    claims = [c.strip("-• ").strip() for c in claims_raw.splitlines() if c.strip()]
    claims = claims[:config.MAX_CLAIMS_TO_VERIFY]

    verdicts = []
    for claim in claims:
        log.append(f"Verifying: {claim[:60]}...")
        evidence = web_search(claim)
        evidence_text = "\n".join(f"- {e['title']}: {e['content']}" for e in evidence)

        verify_prompt = f"""Claim: {claim}

Evidence from web search:
{evidence_text}

Based ONLY on the evidence, is this claim VERIFIED, UNVERIFIED, or CONTRADICTED?
Respond in EXACTLY this format:
VERDICT: <VERIFIED|UNVERIFIED|CONTRADICTED>
NOTE: <one short sentence explaining why>
"""
        result = llm.invoke(verify_prompt).content
        verdict, note = "UNVERIFIED", "Could not determine from available evidence."
        for line in result.splitlines():
            if line.upper().startswith("VERDICT"):
                verdict = line.split(":", 1)[1].strip()
            if line.upper().startswith("NOTE"):
                note = line.split(":", 1)[1].strip()

        verdicts.append({"claim": claim, "verdict": verdict, "note": note})

    log.append("Claim verification complete.")

    # Assemble the final report with a sources + verification appendix
    sources_md = "\n".join(
        f"- [{s['label']}] {s['source']}" for s in state["sources"]
    ) or "- (no external sources used)"

    verdicts_md = "\n".join(
        f"- **{v['verdict']}** — {v['claim']}  \n  _{v['note']}_" for v in verdicts
    )

    final_report = f"""{state['draft']}

---

### Sources
{sources_md}

### Claim Verification
{verdicts_md}
"""

    return {
        **state,
        "claims": claims,
        "verdicts": verdicts,
        "final_report": final_report,
        "log": log,
    }


# ------------------------------------------------------------------
# Build and compile the graph
# ------------------------------------------------------------------
def build_graph():
    graph = StateGraph(ResearchState)

    graph.add_node("gather_context", gather_context)
    graph.add_node("write_draft", write_draft)
    graph.add_node("critique_draft", critique_draft)
    graph.add_node("refine_draft", refine_draft)
    graph.add_node("verify_claims", verify_claims)

    graph.set_entry_point("gather_context")
    graph.add_edge("gather_context", "write_draft")
    graph.add_edge("write_draft", "critique_draft")

    graph.add_conditional_edges(
        "critique_draft",
        should_refine,
        {"refine": "refine_draft", "verify": "verify_claims"},
    )
    graph.add_edge("refine_draft", "critique_draft")  # loop back
    graph.add_edge("verify_claims", END)

    return graph.compile()


def _initial_state(topic: str, use_local: bool = True) -> ResearchState:
    return {
        "topic": topic,
        "use_local": use_local,
        "context": "",
        "sources": [],
        "used_web": False,
        "draft": "",
        "critique_score": 0.0,
        "critique_feedback": "",
        "iteration": 0,
        "claims": [],
        "verdicts": [],
        "final_report": "",
        "log": [],
    }


def run_research(topic: str, use_local: bool = True) -> ResearchState:
    """Runs the whole pipeline in one go and returns only the final state."""
    app = build_graph()
    return app.invoke(_initial_state(topic, use_local))


def stream_research(topic: str, use_local: bool = True):
    """
    Same pipeline, but yields (node_name, state_so_far) after EVERY node
    finishes, instead of waiting for the whole thing to end. This is what
    the Streamlit app uses to show live progress instead of a final dump.
    """
    app = build_graph()
    state = _initial_state(topic, use_local)

    for update in app.stream(state, stream_mode="updates"):
        node_name, node_output = next(iter(update.items()))
        state.update(node_output)
        yield node_name, state

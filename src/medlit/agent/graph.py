import os
import re
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from groq import Groq
from dotenv import load_dotenv

from src.medlit.retrieve.hybrid import HybridRetriever
from src.medlit.retrieve.rerank import Reranker
from src.medlit.generate.llm import generate_answer

load_dotenv()

_RETRIEVER = None
_RERANKER = None
_GROQ_CLIENT = None
REWRITE_MODEL = os.getenv("GROQ_REWRITE_MODEL", "openai/gpt-oss-20b")


def get_retriever():
    global _RETRIEVER
    if _RETRIEVER is None:
        _RETRIEVER = HybridRetriever()
    return _RETRIEVER


def get_reranker():
    global _RERANKER
    if _RERANKER is None:
        _RERANKER = Reranker()
    return _RERANKER


def get_groq_client():
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        _GROQ_CLIENT = Groq(api_key=os.getenv("GROQ_API_KEY"))
    return _GROQ_CLIENT


class AgentState(TypedDict):
    query: str
    chat_history: List[Dict[str, str]]
    search_query: str
    hybrid_results: List[Dict[str, Any]]
    final_chunks: List[Dict[str, Any]]
    answer: str


def strip_html_for_history(text: str) -> str:
    """Strips HTML formatting before passing chat history to the query rewrite LLM."""
    clean = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", clean).strip()


def expand_biomedical_acronyms(query: str) -> str:
    """Expands common biomedical acronyms to optimize full-text and BM25 retrieval."""
    expanded = re.sub(r"\bPD\b", "Parkinson's disease", query, flags=re.IGNORECASE)
    expanded = re.sub(r"\b(a-syn|alpha-syn|asyn)\b", "alpha-synuclein", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\bOLGs?\b", "oligodendrocytes", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\bMSA\b", "multiple system atrophy", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\bDLB\b", "dementia with Lewy bodies", expanded, flags=re.IGNORECASE)
    expanded = re.sub(r"\bCKD\b", "chronic kidney disease", expanded, flags=re.IGNORECASE)
    return expanded


def clean_search_query(text: str) -> str:
    """Removes boolean operators, quotes, and punctuation that degrade embedding/BM25 scores."""
    text = re.sub(r"\b(AND|OR|NOT)\b", " ", text)
    text = re.sub(r"[\"\'\(\)\[\]\{\}]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rewrite_query_node(state: AgentState):
    """Resolves pronouns using chat history and expands biomedical acronyms."""
    raw_query = state.get("query", "").strip()
    expanded_query = expand_biomedical_acronyms(raw_query)

    history = state.get("chat_history", [])[-4:]  # Last 2 turns
    has_pronouns = bool(
        re.search(r"\b(it|they|them|their|this|these|those|that)\b", raw_query, re.IGNORECASE)
    )

    # Fast path: If no conversation history or no pronouns to resolve, use expanded query directly
    if not history or not has_pronouns:
        final_q = clean_search_query(expanded_query) or raw_query
        print(f"--> [Node] Direct Query: '{raw_query}' -> '{final_q}'")
        return {"search_query": final_q}

    history_text = "\n".join(
        f"{m['role'].upper()}: {strip_html_for_history(m['content'])[:250]}"
        for m in history
    )

    prompt = (
        "You are a biomedical search query rewriter.\n"
        "Rewrite the user's latest question into a single natural search query for PubMed/PMC literature retrieval.\n"
        "- Resolve pronouns (e.g. 'it', 'they', 'them') using the conversation history.\n"
        "- Expand acronyms (e.g. 'PD' -> 'Parkinson\\'s disease').\n"
        "- Output ONLY the natural English query without quotes, without boolean operators (no AND/OR), and without preamble.\n\n"
        f"History:\n{history_text}\n\n"
        f"Latest Question: {raw_query}\n"
        "Rewritten Query:"
    )

    try:
        resp = get_groq_client().chat.completions.create(
            model=REWRITE_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500,
        )
        content = resp.choices[0].message.content or ""
        rewritten = clean_search_query(content.strip().strip('"'))
        if not rewritten or len(rewritten) < 3:
            rewritten = clean_search_query(expanded_query)
    except Exception as e:
        print(f"--> [Node] Query rewrite fallback ({e})")
        rewritten = clean_search_query(expanded_query)

    if not rewritten:
        rewritten = clean_search_query(expanded_query) or raw_query

    print(f"--> [Node] Query Rewrite: '{raw_query}' -> '{rewritten}'")
    return {"search_query": rewritten}


def retrieve_node(state: AgentState):
    q = state["search_query"]
    print(f"--> [Node] Retrieving for: {q}")
    results = get_retriever().retrieve(q)
    print(f"    Retrieved {len(results)} hybrid candidates.")
    return {"hybrid_results": results}


def rerank_node(state: AgentState):
    q = state["search_query"]
    print("--> [Node] Reranking chunks...")
    chunks = get_reranker().rerank(q, state["hybrid_results"])
    print(f"    Kept {len(chunks)} chunks after reranking.")
    return {"final_chunks": chunks}


def generate_node(state: AgentState):
    print("--> [Node] Generating grounded answer...")
    # Pass the standalone rewritten query so the LLM also has full context
    answer = generate_answer(state["search_query"], state["final_chunks"])
    return {"answer": answer}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("rewrite", rewrite_query_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)

    workflow.set_entry_point("rewrite")
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()

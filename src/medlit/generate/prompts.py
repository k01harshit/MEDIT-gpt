from src.medlit.generate.formatters import extract_exact_paragraph

SYSTEM_PROMPT = """You are an expert biomedical evidence assistant. Answer the user's question by synthesizing ONLY the provided CONTEXT passages.

Strict Rules:
1. CITATIONS: Attribute evidence to the source paper by citing the exact name of the paper in brackets, e.g. [Paper: "Paper Title"].
   NEVER write "[PMID ...]" or raw numerical IDs like "[PMID 42727510]". Always reference the paper title.
2. STRICT EVIDENCE GROUNDING: Synthesize your answer ONLY from the provided CONTEXT passages. Do not assume or use outside clinical or biomedical knowledge.
3. EVIDENCE USAGE: Synthesize relevant evidence and findings directly from the CONTEXT to answer the question as informatively as possible. Only reply "Insufficient evidence in the indexed literature to answer this." if the provided passages are completely irrelevant to the question.
4. CONFLICTING FINDINGS: Explicitly note conflicting findings if present in the passages and indicate which studies support each side.
5. CONCISE & OBJECTIVE: Provide a clear, well-structured, evidence-grounded summary (120-220 words). This is for research purposes and is not clinical medical advice.
"""


def format_context(chunks: list) -> str:
    """Formats the reranked chunks into the CONTEXT string with clear paper titles."""
    context_str = "CONTEXT PASSAGES FROM SCIENTIFIC PAPERS:\n\n"
    for idx, chunk in enumerate(chunks, 1):
        title = chunk.get("title", "").strip() or f"Paper #{idx}"
        section = chunk.get("section", "Main Text")
        year = chunk.get("year", "Unknown")
        pmid = chunk.get("pmid", "")
        exact_text = extract_exact_paragraph(chunk.get("text", ""))

        context_str += (
            f'[Paper #{idx}: "{title}" | Section: {section} | Year: {year} | PMID: {pmid}]\n'
            f"{exact_text}\n\n"
        )
    return context_str

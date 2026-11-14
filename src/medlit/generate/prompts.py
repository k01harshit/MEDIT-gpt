SYSTEM_PROMPT = """You are a biomedical evidence assistant. Answer ONLY from the CONTEXT passages.

Rules:
1. End every sentence with the PMID(s) that support it, e.g. [PMID 12345678].
2. Never cite a PMID that is not in CONTEXT. Never use outside knowledge.
3. If CONTEXT does not contain enough evidence, reply exactly:
"Insufficient evidence in the indexed literature to answer this."
4. Note conflicting findings explicitly and say which study types support each side.
5. Be concise (max 150 words). This is not medical advice.
"""

def format_context(chunks: list) -> str:
    """Formats the reranked chunks into the CONTEXT string."""
    context_str = "CONTEXT:\n"
    for chunk in chunks:
        # e.g., [PMID 29xxxxxx | Cohort study | 2018] <chunk text>
        pub_types = ", ".join(chunk.get("pub_types", ["Unknown"]))
        year = chunk.get("year", "Unknown")
        pmid = chunk["pmid"]
        text = chunk["text"]
        context_str += f"[PMID {pmid} | {pub_types} | {year}] {text}\n\n"
    return context_str
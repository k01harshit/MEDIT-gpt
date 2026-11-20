import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

# Force load the .env file from the root directory
load_dotenv(find_dotenv(), override=True)

# Fast, reliable Groq model
MODEL_NAME = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

CLEAN_SYSTEM_PROMPT = """You are MedLit Copilot, a rigorous Parkinson's disease biomedical research assistant.
Answer the user's question thoroughly using ONLY the provided numbered literature excerpts.

Formatting & Citation Rules:
- Write a detailed, well-structured scientific response using clear bold section headings and concise bullet points.
- If a comparison table improves clarity, keep every cell brief and scannable.
- Cite sources strictly with numbered brackets like [1], [2], or [1, 3].
- NEVER paste full paper titles inline and NEVER use emojis.
- Do not fabricate facts outside the excerpts. If evidence is partial, answer what is supported and note what is missing."""


def format_numbered_context(context_chunks: list) -> str:
    """Formats retrieved chunks with numbered indices for clean inline citations."""
    blocks = []
    for idx, chunk in enumerate(context_chunks, 1):
        title = chunk.get("title", "Untitled")
        pmid = chunk.get("pmid", "N/A")
        year = chunk.get("year", "")
        section = chunk.get("section", "Body")
        text = chunk.get("text", "").strip()
        blocks.append(
            f"[{idx}] Title: {title} ({year}) | PMID: {pmid} | Section: {section}\n{text}"
        )
    return "\n\n---\n\n".join(blocks)


def generate_answer(query: str, context_chunks: list) -> str:
    """Generates a clean, detailed Markdown answer grounded in the retrieved chunks."""
    if not context_chunks:
        return "Insufficient evidence in the indexed literature to answer this."

    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.1,
        max_tokens=2048,
        api_key=os.getenv("GROQ_API_KEY"),
    )

    formatted_context = format_numbered_context(context_chunks)
    user_message = f"LITERATURE EXCERPTS:\n{formatted_context}\n\nQUESTION: {query}"

    messages = [
        SystemMessage(content=CLEAN_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ]

    response = llm.invoke(messages)
    cleaned = (
        response.content.strip()
        .replace("<br />", "\n")
        .replace("<br/>", "\n")
        .replace("<br>", "\n")
    )
    return cleaned
    


def main():
    from src.medlit.retrieve.hybrid import HybridRetriever
    from src.medlit.retrieve.rerank import Reranker

    test_query = "How does alpha-synuclein pathology affect oligodendrocytes and myelination?"

    print("1. Retrieving...")
    retriever = HybridRetriever()
    hybrid_results = retriever.retrieve(test_query)

    print("2. Reranking...")
    reranker = Reranker()
    final_chunks = reranker.rerank(test_query, hybrid_results)

    print("3. Generating Answer (via LangChain + Groq)...\n")
    answer = generate_answer(test_query, final_chunks)

    print("-" * 50)
    print("QUESTION:", test_query)
    print("ANSWER:\n", answer)
    print("-" * 50)


if __name__ == "__main__":
    main()
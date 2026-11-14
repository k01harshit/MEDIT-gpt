import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

from src.medlit.retrieve.hybrid import HybridRetriever
from src.medlit.retrieve.rerank import Reranker
from src.medlit.generate.prompts import SYSTEM_PROMPT, format_context

# Force load the .env file from the root directory
load_dotenv(find_dotenv(), override=True)

# We will use the standard, fast Llama 3.1 model on Groq
MODEL_NAME = "openai/gpt-oss-20b" 

def generate_answer(query: str, context_chunks: list) -> str:
    """Passes the strict prompt and retrieved context to the LLM using LangChain."""
    
    # Initialize the LangChain Groq client
    llm = ChatGroq(
        model=MODEL_NAME,
        temperature=0.0,
        api_key=os.getenv("GROQ_API_KEY") 
    )
    
    formatted_context = format_context(context_chunks)
    user_message = f"{formatted_context}\nQUESTION: {query}"

    # Construct the LangChain message payload
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_message)
    ]
    
    # Generate the response
    response = llm.invoke(messages)
    
    return response.content

def main():
    test_query = "Does metformin increase the risk of lactic acidosis in patients with mild CKD?"
    
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
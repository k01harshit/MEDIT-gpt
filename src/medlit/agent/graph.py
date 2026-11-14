from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END

# Import our custom RAG components
from src.medlit.retrieve.hybrid import HybridRetriever
from src.medlit.retrieve.rerank import Reranker
from src.medlit.generate.llm import generate_answer

# Initialize models globally so they only load into memory once per session
print("Initializing pipeline models...")
RETRIEVER = HybridRetriever()
RERANKER = Reranker()

class AgentState(TypedDict):
    """The state payload passed between graph nodes."""
    query: str
    hybrid_results: List[Dict[str, Any]]
    final_chunks: List[Dict[str, Any]]
    answer: str

def retrieve_node(state: AgentState):
    print("--> [Node] Retrieving context...")
    results = RETRIEVER.retrieve(state["query"])
    return {"hybrid_results": results}

def rerank_node(state: AgentState):
    print("--> [Node] Reranking chunks...")
    chunks = RERANKER.rerank(state["query"], state["hybrid_results"])
    return {"final_chunks": chunks}

def generate_node(state: AgentState):
    print("--> [Node] Generating answer...")
    answer = generate_answer(state["query"], state["final_chunks"])
    return {"answer": answer}

def build_graph():
    """Compiles the nodes and edges into a LangGraph workflow."""
    workflow = StateGraph(AgentState)
    
    # Define the nodes
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("rerank", rerank_node)
    workflow.add_node("generate", generate_node)
    
    # Define the execution flow
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "rerank")
    workflow.add_edge("rerank", "generate")
    workflow.add_edge("generate", END)
    
    return workflow.compile()

def main():
    # Compile the graph
    app = build_graph()
    
    print("\n" + "=" * 55)
    print(" MedLit Copilot - Pharmacovigilance RAG Agent ")
    print(" Type 'quit' or 'exit' to stop.")
    print("=" * 55)
    
    while True:
        try:
            # Get user input
            user_query = input("\nQUESTION: ")
            
            # Check for exit commands
            if user_query.lower().strip() in ['quit', 'exit', 'q']:
                print("\nShutting down cleanly...")
                break
                
            if not user_query.strip():
                continue
                
            # Invoke the graph with the query
            final_state = app.invoke({"query": user_query})
            
            print("\nANSWER:\n" + final_state["answer"])
            print("-" * 55)
            
        except KeyboardInterrupt:
            # Catch Ctrl+C to exit gracefully without Python crash logs
            print("\n\nShutting down cleanly...")
            break

if __name__ == "__main__":
    main()

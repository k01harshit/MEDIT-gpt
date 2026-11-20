import torch
from sentence_transformers import CrossEncoder
from src.medlit.retrieve.hybrid import HybridRetriever

# Blueprint configuration
RERANKER_MODEL = "BAAI/bge-reranker-base"
TOP_K_KEEP = 15
SCORE_THRESHOLD = 0.0  # bge-reranker scores can be negative; > 0 is generally a good match

class Reranker:
    def __init__(self):
        print(f"Loading Cross-Encoder ({RERANKER_MODEL})...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = CrossEncoder(RERANKER_MODEL, device=self.device, max_length=512)

    def rerank(self, query: str, candidates: list, top_k: int = 10, threshold: float = -2.0):
        if not candidates:
            return []   

        # Prepare pairs for the cross-encoder: (query, document_text)
        pairs = [[query, doc["text"]] for doc in candidates]
        
        # Predict relevance scores
        scores = self.model.predict(pairs)
        
        # Attach scores to candidates
        for idx, doc in enumerate(candidates):
            doc["rerank_score"] = float(scores[idx])
            
        # Sort candidates descending by score
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Filter by threshold if strong matches exist
        passed_threshold = [doc for doc in candidates if doc["rerank_score"] > threshold]
        return passed_threshold[:top_k]

def main():
    # 1. Run the Hybrid Retriever from the previous step
    retriever = HybridRetriever()
    test_query = "Does metformin increase the risk of lactic acidosis in patients with mild CKD?"
    
    print(f"\n1. Running Hybrid Search for: '{test_query}'")
    hybrid_results = retriever.retrieve(test_query)
    print(f"   -> Retrieved {len(hybrid_results)} candidates.")

    # 2. Rerank the results
    reranker = Reranker()
    print("\n2. Reranking candidates...")
    
    final_chunks = reranker.rerank(test_query, hybrid_results)
    
    print(f"\nFinal {len(final_chunks)} chunks after reranking (Threshold > {SCORE_THRESHOLD}):")
    for i, res in enumerate(final_chunks):
        print(f"\n[{i+1}] PMID: {res['pmid']} | Rerank Score: {res['rerank_score']:.4f}")
        print(f"Title: {res['title']}")
        print(f"Snippet: {res['text'][:150]}...")

if __name__ == "__main__":
    main()
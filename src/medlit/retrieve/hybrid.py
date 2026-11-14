import pickle
import torch
from pathlib import Path
from qdrant_client import QdrantClient
from transformers import AutoTokenizer, AutoModel
from nltk.tokenize import word_tokenize

# Paths
PROCESSED_DATA_DIR = Path("data/processed")
BM25_INDEX_FILE = PROCESSED_DATA_DIR / "bm25_index.pkl"
QDRANT_DB_PATH = Path("data/qdrant")

COLLECTION_NAME = "medlit_chunks"
QUERY_MODEL_NAME = "ncbi/MedCPT-Query-Encoder"
RRF_K = 60
TOP_K_RETRIEVE = 50

class HybridRetriever:
    def __init__(self):
        print("Initializing Hybrid Retriever...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load MedCPT Query Encoder
        self.tokenizer = AutoTokenizer.from_pretrained(QUERY_MODEL_NAME)
        self.model = AutoModel.from_pretrained(QUERY_MODEL_NAME).to(self.device)
        self.model.eval()
        
        # Connect to Qdrant
        self.qdrant = QdrantClient(path=str(QDRANT_DB_PATH))
        
        # Load BM25 Index
        with open(BM25_INDEX_FILE, "rb") as f:
            bm25_data = pickle.load(f)
            self.bm25 = bm25_data["bm25"]
            self.chunk_mapping = bm25_data["mapping"]

    def embed_query(self, query: str):
        """Embeds the user question using MedCPT-Query-Encoder."""
        with torch.no_grad():
            encoded_input = self.tokenizer(
                [query], truncation=True, padding=True, max_length=512, return_tensors='pt'
            ).to(self.device)
            output = self.model(**encoded_input)
            # Use [CLS] token
            embedding = output.last_hidden_state[:, 0, :].cpu().numpy()[0]
        return embedding.tolist()

    def dense_search(self, query_vector, limit=TOP_K_RETRIEVE):
        """Searches Qdrant using the updated query_points API."""
        search_result = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit
        ).points
        
        # Return list of dicts with chunk_id and payload
        return [{"chunk_id": hit.payload["chunk_id"], "payload": hit.payload} for hit in search_result]
    def sparse_search(self, query: str, limit=TOP_K_RETRIEVE):
        """Searches BM25."""
        tokens = word_tokenize(query.lower())
        # Get scores for all documents
        scores = self.bm25.get_scores(tokens)
        
        # Get top-k indices
        top_n_indices = scores.argsort()[-limit:][::-1]
        
        results = []
        for idx in top_n_indices:
            if scores[idx] > 0: # Only keep non-zero matches
                chunk_data = self.chunk_mapping[idx]
                results.append({"chunk_id": chunk_data["chunk_id"], "payload": chunk_data})
        return results

    def retrieve(self, query: str):
        """Performs hybrid search and fuses with RRF."""
        # 1. Get Dense and Sparse results
        query_vector = self.embed_query(query)
        dense_results = self.dense_search(query_vector)
        sparse_results = self.sparse_search(query)
        
        # 2. Reciprocal Rank Fusion (RRF)
        rrf_scores = {}
        payloads = {}
        
        # Process Dense
        for rank, item in enumerate(dense_results):
            cid = item["chunk_id"]
            payloads[cid] = item["payload"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            
        # Process Sparse
        for rank, item in enumerate(sparse_results):
            cid = item["chunk_id"]
            payloads[cid] = item["payload"]
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
            
        # 3. Sort by RRF score
        fused = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return sorted list of payloads with their fusion score
        final_results = []
        for cid, score in fused:
            doc = payloads[cid]
            doc["rrf_score"] = score
            final_results.append(doc)
            
        return final_results

def main():
    retriever = HybridRetriever()
    test_query = "Does metformin increase the risk of lactic acidosis in patients with mild CKD?"
    print(f"\nSearching for: '{test_query}'\n")
    
    results = retriever.retrieve(test_query)
    
    print(f"Found {len(results)} fused chunks. Top 3:")
    for i, res in enumerate(results[:3]):
        print(f"\n[{i+1}] PMID: {res['pmid']} (RRF Score: {res['rrf_score']:.4f})")
        print(f"Title: {res['title']}")
        # Print snippet
        print(f"Snippet: {res['text'][:150]}...")

if __name__ == "__main__":
    main()
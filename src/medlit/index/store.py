import json
import pickle
import nltk
from pathlib import Path
from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from rank_bm25 import BM25Okapi

# Download NLTK tokenizer data
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import word_tokenize

# Paths
PROCESSED_DATA_DIR = Path("data/processed")
EMBEDDINGS_FILE = PROCESSED_DATA_DIR / "pubmed_embeddings.jsonl"
QDRANT_DB_PATH = Path("data/qdrant")
BM25_INDEX_FILE = PROCESSED_DATA_DIR / "bm25_index.pkl"

COLLECTION_NAME = "medlit_chunks"
VECTOR_SIZE = 768 # MedCPT Article Encoder dimension

def main():
    print(f"Reading embeddings from {EMBEDDINGS_FILE}...")
    if not EMBEDDINGS_FILE.exists():
        print("Error: Embeddings file not found. Run embed.py first.")
        return

    # Load data
    chunks = []
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))
            
    print(f"Loaded {len(chunks)} chunks.")

    # 1. Setup Qdrant (using local disk storage for simplicity)
    print("\n--- Initializing Qdrant ---")
    client = QdrantClient(path=str(QDRANT_DB_PATH))
    
    # Recreate collection to avoid duplicates during testing
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)
        
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )

    # Prepare Qdrant Points
    points = []
    for idx, chunk in enumerate(chunks):
        # We use the integer index as the Qdrant Point ID, and store the chunk_id in the payload
        payload = {
            "chunk_id": chunk["chunk_id"],
            "pmid": chunk["pmid"],
            "title": chunk["title"],
            "year": chunk["year"],
            "journal": chunk["journal"],
            "pub_types": chunk["pub_types"],
            "mesh_terms": chunk["mesh_terms"],
            "text": chunk["text"]
        }
        
        points.append(
            PointStruct(
                id=idx, 
                vector=chunk["embedding"], 
                payload=payload
            )
        )

    # Upload to Qdrant in batches
    print("Uploading vectors to Qdrant...")
    batch_size = 100
    for i in tqdm(range(0, len(points), batch_size)):
        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points[i : i + batch_size]
        )
    print(f"Successfully indexed {len(points)} vectors in Qdrant.")

    # 2. Build BM25 Index
    print("\n--- Building BM25 Index ---")
    tokenized_corpus = []
    chunk_mapping = {}
    
    for idx, chunk in tqdm(enumerate(chunks), total=len(chunks), desc="Tokenizing texts"):
        # Lowercase and tokenize the text
        tokens = word_tokenize(chunk["text"].lower())
        tokenized_corpus.append(tokens)
        
        # Save mapping from internal integer ID to chunk_id and payload
# Update this mapping inside the BM25 loop (around line 84)
        chunk_mapping[idx] = {
            "chunk_id": chunk["chunk_id"],
            "pmid": chunk["pmid"],
            "title": chunk["title"],           # <-- Added
            "year": chunk["year"],             # <-- Added
            "journal": chunk["journal"],       # <-- Added
            "pub_types": chunk["pub_types"],   # <-- Added
            "text": chunk["text"]
        }

    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the BM25 object and mapping to disk
    bm25_data = {
        "bm25": bm25,
        "mapping": chunk_mapping
    }
    
    with open(BM25_INDEX_FILE, "wb") as f:
        pickle.dump(bm25_data, f)
        
    print(f"BM25 index saved to {BM25_INDEX_FILE}")
    print("\nWeek 1 milestone complete! The knowledge base is fully indexed.")

if __name__ == "__main__":
    main()
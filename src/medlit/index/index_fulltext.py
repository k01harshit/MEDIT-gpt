import json
import uuid
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models

CHUNKS_PATH = "data/processed/parkinsons_chunks.jsonl"
QDRANT_PATH = "data/qdrant"
COLLECTION_NAME = "parkinsons_fulltext"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"


def build_qdrant_index(batch_size: int = 64):
    print(f"Loading embedding model: {EMBED_MODEL}...")
    model = SentenceTransformer(EMBED_MODEL)
    vector_size = model.get_sentence_embedding_dimension()

    client = QdrantClient(path=QDRANT_PATH)

    # Recreate collection cleanly for the new full-text Parkinson's corpus
    if client.collection_exists(COLLECTION_NAME):
        client.delete_collection(COLLECTION_NAME)

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=vector_size,
            distance=models.Distance.COSINE,
        ),
    )

    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line))

    print(f"Indexing {len(chunks)} chunks into '{COLLECTION_NAME}'...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c["text"] for c in batch]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()

        points = [
            models.PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_DNS, c["chunk_id"])),
                vector=vec,
                payload={
                    "chunk_id": c["chunk_id"],
                    "pmid": c["pmid"],
                    "pmcid": c.get("pmcid", ""),
                    "topic": c["topic"],
                    "title": c["title"],
                    "year": c["year"],
                    "section": c["section"],
                    "chunk_type": c["chunk_type"],
                    "text": c["text"],
                },
            )
            for c, vec in zip(batch, vectors)
        ]

        client.upsert(collection_name=COLLECTION_NAME, points=points)
        if (i // batch_size) % 10 == 0 or i + batch_size >= len(chunks):
            print(f"  Upserted {min(i + batch_size, len(chunks))}/{len(chunks)}")

    print(f"Done! Collection '{COLLECTION_NAME}' is ready at {QDRANT_PATH}.")


if __name__ == "__main__":
    build_qdrant_index()
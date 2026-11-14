import json
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

# Paths
PROCESSED_DATA_DIR = Path("data/processed")
INPUT_FILE = PROCESSED_DATA_DIR / "pubmed_chunks.jsonl"
OUTPUT_FILE = PROCESSED_DATA_DIR / "pubmed_embeddings.jsonl"

# Blueprint: MedCPT Article Encoder (768-d)
MODEL_NAME = "ncbi/MedCPT-Article-Encoder"
BATCH_SIZE = 32

def main():
    print("Loading MedCPT Article Encoder...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # Read all chunks
    print(f"Reading chunks from {INPUT_FILE}...")
    if not INPUT_FILE.exists():
        print("Error: Chunk file not found. Run chunking.py first.")
        return

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f]
        
    print(f"Loaded {len(chunks)} chunks. Starting embedding generation...")
    
    # Process in batches
    if OUTPUT_FILE.exists():
        OUTPUT_FILE.unlink()
        
    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        for i in tqdm(range(0, len(chunks), BATCH_SIZE)):
            batch = chunks[i : i + BATCH_SIZE]
            texts = [item["text"] for item in batch]
            
            # Tokenize and encode
            with torch.no_grad():
                encoded_input = tokenizer(
                    texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(device)
                
                output = model(**encoded_input)
                # MedCPT uses the [CLS] token representation as the sentence embedding
                embeddings = output.last_hidden_state[:, 0, :].cpu().numpy()
                
            # Save original metadata plus the embedding
            for idx, item in enumerate(batch):
                item["embedding"] = embeddings[idx].tolist()
                out_f.write(json.dumps(item) + "\n")

    print(f"Finished! Embeddings saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
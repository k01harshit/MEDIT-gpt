import json
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Paths
RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Blueprint parameters: ~350 tokens, 50 overlap
chunk_size = 350
chunk_overlap = 50

# Initialize the text splitter (using tiktoken for approximate token counting)
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", ".", " ", ""] # Respects paragraphs/sections first
)

def process_file(file_path, out_file):
    """Reads raw JSONL, chunks abstracts, and writes to processed JSONL."""
    chunks_created = 0
    with open(file_path, "r", encoding="utf-8") as infile, \
         open(out_file, "a", encoding="utf-8") as outfile:
        
        for line in infile:
            if not line.strip():
                continue
                
            article = json.loads(line)
            pmid = article["pmid"]
            title = article["title"]
            abstract = article.get("abstract", "")
            
            if not abstract:
                continue
                
            # Split the abstract into chunks
            raw_chunks = text_splitter.split_text(abstract)
            
            for i, chunk_text in enumerate(raw_chunks):
                # Prepend the title to each chunk so context is retained
                enriched_text = f"Title: {title}\n{chunk_text}"
                
                chunk_record = {
                    "chunk_id": f"{pmid}_{i}",
                    "pmid": pmid,
                    "title": title,
                    "year": article["year"],
                    "journal": article["journal"],
                    "pub_types": article["pub_types"],
                    "mesh_terms": article["mesh_terms"],
                    "text": enriched_text
                }
                
                outfile.write(json.dumps(chunk_record) + "\n")
                chunks_created += 1
                
    return chunks_created

def main():
    print("Starting section-aware chunking...")
    out_file = PROCESSED_DATA_DIR / "pubmed_chunks.jsonl"
    
    # Clear the output file if it already exists
    if out_file.exists():
        out_file.unlink()
        
    total_chunks = 0
    # Process all batch files in the raw directory
    for raw_file in RAW_DATA_DIR.glob("pubmed_batch_*.jsonl"):
        print(f"Processing {raw_file.name}...")
        chunks = process_file(raw_file, out_file)
        total_chunks += chunks
        
    print(f"Finished! Created {total_chunks} chunks saved to {out_file}")

if __name__ == "__main__":
    main()
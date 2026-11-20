import os
import json
from langchain_text_splitters import RecursiveCharacterTextSplitter

INPUT_JSONL = "data/processed/parkinsons_fulltext.jsonl"
OUTPUT_CHUNKS = "data/processed/parkinsons_chunks.jsonl"


def chunk_fulltext_corpus(chunk_size: int = 1000, chunk_overlap: int = 150):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    os.makedirs(os.path.dirname(OUTPUT_CHUNKS), exist_ok=True)
    total_papers = 0
    total_chunks = 0

    with open(INPUT_JSONL, "r", encoding="utf-8") as in_f, open(
        OUTPUT_CHUNKS, "w", encoding="utf-8"
    ) as out_f:
        for line in in_f:
            paper = json.loads(line)
            total_papers += 1
            base_meta = {
                "pmid": paper.get("pmid", "Unknown"),
                "pmcid": paper.get("pmcid", ""),
                "topic": paper.get("topic", "Parkinson's"),
                "title": paper.get("title", "Untitled"),
                "year": paper.get("year", "Unknown"),
            }

            # 1. Abstract chunk
            abstract = paper.get("abstract", "").strip()
            if abstract:
                for idx, chunk_text in enumerate(splitter.split_text(abstract)):
                    record = {
                        **base_meta,
                        "section": "Abstract",
                        "chunk_type": "abstract",
                        "chunk_id": f"{base_meta['pmid']}_abs_{idx}",
                        "text": f"[{base_meta['title']} | Abstract]\n{chunk_text}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

            # 2. Body section chunks
            for s_idx, sec in enumerate(paper.get("sections", [])):
                heading = sec.get("heading", "Body")
                sec_text = sec.get("text", "").strip()
                if not sec_text:
                    continue
                for c_idx, chunk_text in enumerate(splitter.split_text(sec_text)):
                    record = {
                        **base_meta,
                        "section": heading,
                        "chunk_type": "section",
                        "chunk_id": f"{base_meta['pmid']}_sec{s_idx}_{c_idx}",
                        "text": f"[{base_meta['title']} | {heading}]\n{chunk_text}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

            # 3. Table chunks
            for t_idx, tbl in enumerate(paper.get("tables", [])):
                caption = tbl.get("caption", f"Table {t_idx + 1}")
                content = tbl.get("content", "").strip()
                if not content:
                    continue
                table_text = f"Table: {caption}\n{content}"
                for c_idx, chunk_text in enumerate(splitter.split_text(table_text)):
                    record = {
                        **base_meta,
                        "section": caption,
                        "chunk_type": "table",
                        "chunk_id": f"{base_meta['pmid']}_tbl{t_idx}_{c_idx}",
                        "text": f"[{base_meta['title']} | {caption}]\n{chunk_text}",
                    }
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_chunks += 1

    print(f"Chunked {total_papers} papers into {total_chunks} chunks -> {OUTPUT_CHUNKS}")


if __name__ == "__main__":
    chunk_fulltext_corpus()
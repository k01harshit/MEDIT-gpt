import os
import json
import time
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL", "research@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")

FULLTEXT_JSONL = "data/processed/parkinsons_fulltext.jsonl"


def patch_journal_venues(batch_size: int = 100):
    with open(FULLTEXT_JSONL, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    # Collect numeric PMIDs
    pmids = [str(r.get("pmid", "")).strip() for r in records if str(r.get("pmid", "")).isdigit()]
    venue_map = {}

    print(f"Fetching real journal venues from PubMed for {len(pmids)} papers...")
    for i in range(0, len(pmids), batch_size):
        batch = pmids[i : i + batch_size]
        handle = Entrez.esummary(db="pubmed", id=",".join(batch))
        summaries = Entrez.read(handle)
        handle.close()

        for doc in summaries:
            pid = str(doc.get("Id", "")).strip()
            journal = str(doc.get("FullJournalName") or doc.get("Source") or "").strip()
            if pid and journal:
                venue_map[pid] = journal

        print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)}")
        time.sleep(0.15)

    updated = 0
    for r in records:
        pid = str(r.get("pmid", "")).strip()
        if pid in venue_map:
            r["journal"] = venue_map[pid]
            updated += 1

    with open(FULLTEXT_JSONL, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Done! Updated real journal venues for {updated}/{len(records)} papers.")


if __name__ == "__main__":
    patch_journal_venues()
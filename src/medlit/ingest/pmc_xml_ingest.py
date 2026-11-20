import os
import json
import time
from Bio import Entrez
from lxml import etree
from dotenv import load_dotenv

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL", "research@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")

OUTPUT_JSONL = "data/processed/parkinsons_fulltext.jsonl"

# Topic queries and target quotas (total = 500 papers)
TOPIC_QUOTAS = [
    (
        "Oligodendrocytes",
        '("Parkinson\'s disease"[Title/Abstract] OR "Parkinson disease"[MeSH]) AND ("oligodendrocyte"[Title/Abstract] OR "oligodendrocytes"[Title/Abstract]) AND "open access"[filter]',
        100,
    ),
    (
        "Myelination",
        '("Parkinson\'s disease"[Title/Abstract] OR "Parkinson disease"[MeSH]) AND ("myelin"[Title/Abstract] OR "myelination"[Title/Abstract] OR "demyelination"[Title/Abstract]) AND "open access"[filter]',
        100,
    ),
    (
        "Alpha-Synuclein",
        '("Parkinson\'s disease"[Title/Abstract] OR "Parkinson disease"[MeSH]) AND "alpha-synuclein"[Title/Abstract] AND "open access"[filter]',
        100,
    ),
    (
        "Biomarkers",
        '("Parkinson\'s disease"[Title/Abstract] OR "Parkinson disease"[MeSH]) AND ("biomarker"[Title/Abstract] OR "biomarkers"[Title/Abstract]) AND "open access"[filter]',
        100,
    ),
    (
        "Landmark Parkinson's",
        '("Parkinson\'s disease"[Title] OR "Parkinson disease"[Title]) AND ("review"[Publication Type] OR "clinical trial"[Publication Type]) AND "open access"[filter]',
        100,
    ),
]


def search_pmc_ids(query: str, retmax: int) -> list[str]:
    """Fetch PMC IDs sorted by relevance."""
    handle = Entrez.esearch(db="pmc", term=query, retmax=retmax, sort="relevance")
    result = Entrez.read(handle)
    handle.close()
    return result.get("IdList", [])


def parse_jats_article(article_el, topic: str) -> dict | None:
    """Extract metadata, abstract, body sections, and tables from a PMC JATS XML <article>."""
    # IDs
    pmid = "".join(article_el.xpath(".//article-meta/article-id[@pub-id-type='pmid']/text()")).strip()
    pmcid = "".join(article_el.xpath(".//article-meta/article-id[@pub-id-type='pmc']/text()")).strip()
    if pmcid and not pmcid.startswith("PMC"):
        pmcid = f"PMC{pmcid}"

    # Title
    title_parts = article_el.xpath(".//article-meta/title-group/article-title//text()")
    title = " ".join(t.strip() for t in title_parts if t.strip())
    if not title:
        return None

    # Year
    years = article_el.xpath(".//article-meta/pub-date/year/text()")
    year = years[0].strip() if years else "Unknown"

    # Abstract
    abstract_parts = article_el.xpath(".//article-meta/abstract//text()")
    abstract = " ".join(t.strip() for t in abstract_parts if t.strip())

    # Body sections
    sections = []
    for sec in article_el.xpath(".//body//sec"):
        heading = " ".join(sec.xpath("./title//text()")).strip() or "Section"
        paragraphs = [
            " ".join(p.xpath(".//text()")).strip()
            for p in sec.xpath("./p")
        ]
        body_text = "\n".join(p for p in paragraphs if p)
        if body_text:
            sections.append({"heading": heading, "text": body_text})

    # Tables
    tables = []
    for tw in article_el.xpath(".//table-wrap"):
        caption = " ".join(tw.xpath("./label//text() | ./caption//text()")).strip()
        rows = []
        for tr in tw.xpath(".//table//tr"):
            cells = [" ".join(c.xpath(".//text()")).strip() for c in tr.xpath("./th | ./td")]
            if any(cells):
                rows.append(" | ".join(cells))
        if rows:
            tables.append({"caption": caption, "content": "\n".join(rows)})

    if not abstract and not sections:
        return None

    return {
        "pmid": pmid or pmcid,
        "pmcid": pmcid,
        "topic": topic,
        "title": title,
        "year": year,
        "abstract": abstract,
        "sections": sections,
        "tables": tables,
    }


def fetch_and_parse_batch(pmc_ids: list[str], topic: str) -> list[dict]:
    """Fetch full-text JATS XML for a batch of PMC IDs and parse them."""
    handle = Entrez.efetch(db="pmc", id=",".join(pmc_ids), rettype="xml", retmode="xml")
    xml_bytes = handle.read()
    handle.close()

    parser = etree.XMLParser(recover=True)
    root = etree.fromstring(xml_bytes, parser=parser)

    records = []
    for article_el in root.xpath(".//article"):
        parsed = parse_jats_article(article_el, topic)
        if parsed:
            records.append(parsed)
    return records


def main(batch_size: int = 25):
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    seen_ids = set()
    total_saved = 0

    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for topic_name, query, quota in TOPIC_QUOTAS:
            print(f"\n--- Topic: {topic_name} (target: {quota}) ---")
            # Fetch slightly more IDs to account for deduplication across topics
            candidate_ids = search_pmc_ids(query, retmax=int(quota * 1.5))
            unique_ids = [cid for cid in candidate_ids if cid not in seen_ids][:quota]
            seen_ids.update(unique_ids)
            print(f"Found {len(unique_ids)} unique Open Access PMC articles.")

            for i in range(0, len(unique_ids), batch_size):
                batch = unique_ids[i : i + batch_size]
                try:
                    records = fetch_and_parse_batch(batch, topic_name)
                    for rec in records:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    total_saved += len(records)
                    print(f"  Parsed batch {i // batch_size + 1} -> {len(records)} papers (Total: {total_saved})")
                except Exception as e:
                    print(f"  Batch error ({topic_name}, idx {i}): {e}")
                time.sleep(0.3)

    print(f"\nDone! Saved {total_saved} full-text Parkinson's papers to {OUTPUT_JSONL}")


if __name__ == "__main__":
    main()
import os
import json
import time
import requests
from Bio import Entrez
from lxml import etree
from dotenv import load_dotenv

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL", "research@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")

GROBID_URL = "http://localhost:8070/api/processFulltextDocument"
PMC_OA_API = "https://www.ncbi.nlm.nih.gov/pmc/utils/oa/oa.fcgi"
PDF_DIR = "data/raw/pdfs"
OUTPUT_JSONL = "data/processed/parkinsons_fulltext.jsonl"

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}


def search_pmc_oa_parkinsons(max_papers: int = 25) -> list[dict]:
    """Search PMC for Open Access Parkinson's articles and fetch basic metadata."""
    term = '("Parkinson Disease"[MeSH] OR "Parkinson\'s disease"[Title/Abstract]) AND "open access"[filter]'
    handle = Entrez.esearch(db="pmc", term=term, retmax=max_papers * 2, sort="relevance")
    id_list = Entrez.read(handle)["IdList"]
    handle.close()

    if not id_list:
        return []

    # Fetch titles, years, and PMIDs via esummary
    summary_handle = Entrez.esummary(db="pmc", id=",".join(id_list))
    summaries = Entrez.read(summary_handle)
    summary_handle.close()

    papers = []
    for doc in summaries:
        pmcid = f"PMC{doc['Id']}"
        article_ids = doc.get("ArticleIds", {})
        pmid = article_ids.get("pmid", pmcid)
        pub_date = str(doc.get("PubDate", "Unknown"))
        year = pub_date.split()[0] if pub_date else "Unknown"
        papers.append({
            "pmcid": pmcid,
            "pmid": pmid,
            "title": doc.get("Title", "Untitled"),
            "year": year,
        })
    return papers


def get_official_pdf_url(pmcid: str) -> str | None:
    """Query NCBI PMC OA Web Service for the direct FTP/HTTPS PDF link."""
    resp = requests.get(PMC_OA_API, params={"id": pmcid}, timeout=20)
    if resp.status_code != 200:
        return None
    root = etree.fromstring(resp.content)
    pdf_links = root.xpath("//link[@format='pdf']/@href")
    if not pdf_links:
        return None
    # Convert ftp:// to https:// for direct requests download
    return pdf_links[0].replace("ftp://", "https://")


def download_pdf(pmcid: str, pdf_url: str) -> str | None:
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_path = os.path.join(PDF_DIR, f"{pmcid}.pdf")
    if os.path.exists(pdf_path):
        return pdf_path

    try:
        resp = requests.get(pdf_url, timeout=60)
        if resp.status_code == 200 and resp.content[:4] == b"%PDF":
            with open(pdf_path, "wb") as f:
                f.write(resp.content)
            return pdf_path
    except requests.RequestException as e:
        print(f"  -> Download error: {e}")
    return None


def parse_pdf_with_grobid(pdf_path: str) -> dict | None:
    with open(pdf_path, "rb") as f:
        files = {"input": (os.path.basename(pdf_path), f, "application/pdf")}
        resp = requests.post(GROBID_URL, files=files, timeout=120)

    if resp.status_code != 200:
        return None

    root = etree.fromstring(resp.content)

    abstract_parts = root.xpath("//tei:profileDesc/tei:abstract//text()", namespaces=TEI_NS)
    abstract = " ".join(t.strip() for t in abstract_parts if t.strip())

    sections = []
    for div in root.xpath("//tei:body/tei:div", namespaces=TEI_NS):
        head = " ".join(div.xpath("./tei:head//text()", namespaces=TEI_NS)).strip()
        paragraphs = [
            " ".join(p.xpath(".//text()")).strip()
            for p in div.xpath("./tei:p", namespaces=TEI_NS)
        ]
        body_text = "\n".join(p for p in paragraphs if p)
        if body_text:
            sections.append({"heading": head or "Body", "text": body_text})

    tables = []
    for fig in root.xpath("//tei:figure[@type='table']", namespaces=TEI_NS):
        caption = " ".join(fig.xpath("./tei:head//text() | ./tei:figDesc//text()", namespaces=TEI_NS)).strip()
        rows = []
        for row in fig.xpath(".//tei:row", namespaces=TEI_NS):
            cells = [" ".join(c.xpath(".//text()")).strip() for c in row.xpath("./tei:cell", namespaces=TEI_NS)]
            rows.append(" | ".join(cells))
        if rows:
            tables.append({"caption": caption, "content": "\n".join(rows)})

    return {"abstract": abstract, "sections": sections, "tables": tables}


def main(target_papers: int = 20):
    os.makedirs(os.path.dirname(OUTPUT_JSONL), exist_ok=True)
    papers = search_pmc_oa_parkinsons(max_papers=target_papers)
    print(f"Found {len(papers)} PMC Open Access Parkinson's candidates.")

    processed_count = 0
    with open(OUTPUT_JSONL, "w", encoding="utf-8") as out_f:
        for paper in papers:
            if processed_count >= target_papers:
                break

            pmcid = paper["pmcid"]
            pdf_url = get_official_pdf_url(pmcid)
            if not pdf_url:
                continue

            print(f"[{processed_count + 1}/{target_papers}] Downloading {pmcid}...")
            pdf_path = download_pdf(pmcid, pdf_url)
            if not pdf_path:
                continue

            print("  -> Parsing with GROBID...")
            parsed = parse_pdf_with_grobid(pdf_path)
            if not parsed:
                print(f"  -> GROBID failed on {pmcid}")
                continue

            record = {
                "pmid": paper["pmid"],
                "pmcid": pmcid,
                "title": paper["title"],
                "year": paper["year"],
                "abstract": parsed["abstract"],
                "sections": parsed["sections"],
                "tables": parsed["tables"],
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            processed_count += 1
            time.sleep(0.3)

    print(f"\nDone! Parsed {processed_count} full-text papers into {OUTPUT_JSONL}")


if __name__ == "__main__":
    main(target_papers=20)
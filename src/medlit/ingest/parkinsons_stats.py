import os
import json
import time
import pandas as pd
from Bio import Entrez
from dotenv import load_dotenv

load_dotenv()
Entrez.email = os.getenv("NCBI_EMAIL", "research@example.com")
Entrez.api_key = os.getenv("NCBI_API_KEY")

PD_BASE = '("Parkinson Disease"[MeSH] OR "Parkinson\'s disease"[Title/Abstract])'

# Comprehensive Parkinson's disease taxonomy for the all-time yearly dashboard
ALL_PD_TOPICS = {
    "Overall Parkinson's": PD_BASE,
    "Alpha-Synuclein & Lewy Pathology": f'{PD_BASE} AND ("alpha-synuclein"[Title/Abstract] OR "Lewy bodies"[Title/Abstract] OR "alpha-Synuclein"[MeSH])',
    "Genetics (LRRK2/GBA/PINK1/PRKN)": f'{PD_BASE} AND ("LRRK2"[Title/Abstract] OR "GBA"[Title/Abstract] OR "PINK1"[Title/Abstract] OR "SNCA"[Title/Abstract] OR "genetics"[MeSH Subheading])',
    "Pharmacotherapy & Levodopa": f'{PD_BASE} AND ("levodopa"[Title/Abstract] OR "dopamine agonist"[Title/Abstract] OR "MAO-B"[Title/Abstract] OR "pharmacotherapy"[Title/Abstract])',
    "Deep Brain Stimulation & Surgery": f'{PD_BASE} AND ("deep brain stimulation"[Title/Abstract] OR "DBS"[Title/Abstract] OR "subthalamic nucleus"[Title/Abstract])',
    "Biomarkers & Diagnostics": f'{PD_BASE} AND ("biomarker"[Title/Abstract] OR "biomarkers"[Title/Abstract] OR "Biomarkers"[MeSH])',
    "Neuroinflammation & Glia": f'{PD_BASE} AND ("neuroinflammation"[Title/Abstract] OR "microglia"[Title/Abstract] OR "astrocyte"[Title/Abstract] OR "oligodendrocyte"[Title/Abstract])',
    "Myelination & White Matter": f'{PD_BASE} AND ("myelin"[Title/Abstract] OR "myelination"[Title/Abstract] OR "demyelination"[Title/Abstract] OR "white matter"[Title/Abstract])',
    "Mitochondria & Oxidative Stress": f'{PD_BASE} AND ("mitochondria"[Title/Abstract] OR "mitochondrial"[Title/Abstract] OR "oxidative stress"[Title/Abstract])',
    "Non-Motor Symptoms & Cognition": f'{PD_BASE} AND ("non-motor"[Title/Abstract] OR "dementia"[Title/Abstract] OR "cognitive impairment"[Title/Abstract] OR "REM sleep"[Title/Abstract])',
    "Gut-Brain Axis & Microbiome": f'{PD_BASE} AND ("gut-brain"[Title/Abstract] OR "microbiome"[Title/Abstract] OR "microbiota"[Title/Abstract] OR "gastrointestinal"[Title/Abstract])',
    "AI, Machine Learning & Neuroimaging": f'{PD_BASE} AND ("machine learning"[Title/Abstract] OR "deep learning"[Title/Abstract] OR "MRI"[Title/Abstract] OR "PET"[Title/Abstract])',
    "Stem Cells & Gene Therapy": f'{PD_BASE} AND ("stem cell"[Title/Abstract] OR "iPSC"[Title/Abstract] OR "gene therapy"[Title/Abstract] OR "cell transplantation"[Title/Abstract])',
}


def fetch_yearly_counts(start_year: int = 2000, end_year: int = 2026):
    records = []
    for topic_name, query in ALL_PD_TOPICS.items():
        print(f"Fetching yearly counts: {topic_name}...")
        for year in range(start_year, end_year + 1):
            term = f"({query}) AND ({year}[PDAT])"
            handle = Entrez.esearch(db="pubmed", term=term, retmax=0)
            result = Entrez.read(handle)
            handle.close()

            records.append({
                "Topic": topic_name,
                "Year": year,
                "Paper_Count": int(result["Count"]),
            })
            time.sleep(0.12)

    os.makedirs("data/processed", exist_ok=True)
    df = pd.DataFrame(records)
    df.to_csv("data/processed/parkinsons_yearly_stats.csv", index=False)
    print(f"Saved {len(df)} yearly rows -> data/processed/parkinsons_yearly_stats.csv")


def fetch_pd_abstracts(max_abstracts: int = 2000, batch_size: int = 200):
    """Fetch Parkinson's papers with metadata + abstracts for dashboard exploration."""
    print(f"\nFetching {max_abstracts} Parkinson's abstracts from PubMed...")
    handle = Entrez.esearch(db="pubmed", term=PD_BASE, retmax=max_abstracts, sort="relevance")
    pmids = Entrez.read(handle)["IdList"]
    handle.close()

    os.makedirs("data/processed", exist_ok=True)
    out_path = "data/processed/parkinsons_abstracts_meta.jsonl"
    saved = 0

    with open(out_path, "w", encoding="utf-8") as out_f:
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i : i + batch_size]
            fetch_handle = Entrez.efetch(db="pubmed", id=",".join(batch), retmode="xml")
            papers = Entrez.read(fetch_handle)
            fetch_handle.close()

            for article in papers.get("PubmedArticle", []):
                medline = article["MedlineCitation"]
                art = medline["Article"]
                pmid = str(medline["PMID"])
                title = str(art.get("ArticleTitle", ""))
                journal = str(art.get("Journal", {}).get("Title", ""))

                pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
                year = str(pub_date.get("Year") or pub_date.get("MedlineDate", "Unknown"))[:4]

                abs_list = art.get("Abstract", {}).get("AbstractText", [])
                abstract = " ".join(str(x) for x in abs_list).strip()
                if not abstract:
                    continue

                out_f.write(json.dumps({
                    "pmid": pmid,
                    "title": title,
                    "journal": journal,
                    "year": year,
                    "abstract": abstract,
                }, ensure_ascii=False) + "\n")
                saved += 1

            print(f"  Fetched {min(i + batch_size, len(pmids))}/{len(pmids)} (saved {saved} with abstracts)")
            time.sleep(0.2)

    print(f"Saved {saved} abstracts -> {out_path}")


if __name__ == "__main__":
    fetch_yearly_counts(start_year=2000, end_year=2026)
    fetch_pd_abstracts(max_abstracts=2000)
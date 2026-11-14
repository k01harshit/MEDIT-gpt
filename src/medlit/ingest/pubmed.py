import os
import json
import yaml
from pathlib import Path
from dotenv import load_dotenv
from Bio import Entrez
from tenacity import retry, wait_exponential, stop_after_attempt

# Load environment variables
load_dotenv()

# Configure Entrez
Entrez.email = "kharshit4870@gmail.com"  # Using your known email
Entrez.api_key = os.getenv("NCBI_API_KEY")

# Paths
CONFIG_PATH = Path("configs/ingest.yaml")
RAW_DATA_DIR = Path("data/raw")
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def search_pubmed(query, year_from, year_to):
    """Search PubMed and return the history server WebEnv and QueryKey."""
    print(f"Searching PubMed for: {query}")
    
    # Optional: you can append the date range to the query
    date_filter = f' AND ("{year_from}/01/01"[Date - Publication] : "{year_to}/12/31"[Date - Publication])'
    full_query = query + date_filter
    
    handle = Entrez.esearch(
        db="pubmed",
        term=full_query,
        usehistory="y" # This is critical: saves the search history on the server
    )
    record = Entrez.read(handle)
    handle.close()
    return record["Count"], record["QueryKey"], record["WebEnv"]

@retry(wait=wait_exponential(multiplier=1, min=4, max=60), stop=stop_after_attempt(5))
def fetch_details(query_key, webenv, retstart, batch_size=200):
    """Fetch a batch of XML records from the history server."""
    print(f"Fetching records {retstart} to {retstart + batch_size}...")
    handle = Entrez.efetch(
        db="pubmed",
        retmode="xml",
        retstart=retstart,
        retmax=batch_size,
        webenv=webenv,
        query_key=query_key
    )
    records = Entrez.read(handle)
    handle.close()
    return records

def parse_record(article_data):
    """Extract required fields from the raw XML dictionary."""
    try:
        medline = article_data.get("MedlineCitation", {})
        article = medline.get("Article", {})
        
        pmid = str(medline.get("PMID", ""))
        title = article.get("ArticleTitle", "")
        journal = article.get("Journal", {}).get("Title", "")
        
        # Get Year safely
        pub_date = article.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
        year = pub_date.get("Year", "")
        
        # Get Publication Types
        pub_types = [pt.title() for pt in article.get("PublicationTypeList", [])]
        
        # Get MeSH Terms
        mesh_headings = medline.get("MeshHeadingList", [])
        mesh_terms = [mesh["DescriptorName"].title() for mesh in mesh_headings]
        
        # Parse Abstract Sections
        abstract = article.get("Abstract", {})
        abstract_text = abstract.get("AbstractText", [])
        
        sections = []
        if isinstance(abstract_text, list):
            for section in abstract_text:
                label = getattr(section, "attributes", {}).get("Label", "TEXT")
                sections.append(f"{label}: {str(section)}")
        else:
            sections.append(str(abstract_text))
            
        full_abstract = "\n".join(sections)
        
        return {
            "pmid": pmid,
            "title": title,
            "journal": journal,
            "year": year,
            "pub_types": pub_types,
            "mesh_terms": mesh_terms,
            "abstract": full_abstract
        }
    except Exception as e:
        print(f"Error parsing record: {e}")
        return None

def main():
    config = load_config()
    search_cfg = config["search"]
    
    count, query_key, webenv = search_pubmed(
        search_cfg["query"], 
        search_cfg["year_from"], 
        search_cfg["year_to"]
    )
    
    count = int(count)
    print(f"Found {count} articles. Starting ingestion...")
    
    batch_size = 200
    # For testing, let's limit to the first 400 articles to make sure it works.
    # To download all ~40k, change the limit or remove `min(count, 400)` and use `count`.
    limit = min(count, 400) 
    
    for start in range(0, limit, batch_size):
        records = fetch_details(query_key, webenv, start, batch_size)
        
        parsed_batch = []
        for article in records.get("PubmedArticle", []):
            parsed = parse_record(article)
            if parsed and parsed["abstract"].strip():
                parsed_batch.append(parsed)
                
        # Save batch to jsonl
        out_file = RAW_DATA_DIR / f"pubmed_batch_{start}.jsonl"
        with open(out_file, "w", encoding="utf-8") as f:
            for item in parsed_batch:
                f.write(json.dumps(item) + "\n")
                
        print(f"Saved {len(parsed_batch)} parsed articles to {out_file}")

if __name__ == "__main__":
    main()
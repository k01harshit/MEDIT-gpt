import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.medlit.generate.formatters import format_rag_response, PAPER_PALETTE
from src.medlit.generate.prompts import SYSTEM_PROMPT, format_context

def test_citations_and_colors():
    chunks = [
        {
            "pmid": "42727510",
            "title": "Alpha-Synuclein Pathology in Parkinsonian Glia",
            "section": "Results",
            "year": "2024",
            "text": "[Alpha-Synuclein Pathology in Parkinsonian Glia | Results]\nAlpha-synuclein aggregates trigger massive demyelination and selective loss of myelin-forming oligodendroglia in model systems."
        },
        {
            "pmid": "37386505",
            "title": "The contribution of DNA methylation to the (dys)function of oligodendroglia in neurodegeneration",
            "section": "Abstract",
            "year": "2023",
            "text": "[The contribution of DNA methylation to the (dys)function of oligodendroglia in neurodegeneration | Abstract]\nAberrant DNA methylation has been associated with Parkinson's disease and highlights epigenetic changes in oligodendrocyte genes."
        }
    ]

    # Test 1: Replace raw [PMID 42727510]
    raw_answer_1 = "Alpha-synuclein aggregation leads to demyelination [PMID 42727510]."
    formatted_1 = format_rag_response(raw_answer_1, chunks)
    assert "[PMID 42727510" not in formatted_1, "PMID 42727510 should not be visible"
    assert "Alpha-Synuclein Pathology in Parkinsonian Glia" in formatted_1
    assert "rag-exact-container" in formatted_1
    assert "rag-summary-container" in formatted_1
    assert "Exact Paragraph(s) Written in Paper" in formatted_1
    assert "Summarized Answer" in formatted_1
    print("Test 1 Passed: [PMID 42727510] successfully replaced with paper title.")

    # Test 2: Multi-PMID in one bracket
    raw_answer_2 = "Both findings support oligodendrocyte vulnerability [PMID 42727510, PMID 37386505]."
    formatted_2 = format_rag_response(raw_answer_2, chunks)
    assert "[PMID" not in formatted_2
    assert "Alpha-Synuclein Pathology in Parkinsonian Glia" in formatted_2
    assert "The contribution of DNA methylation to the (dys)function of oligodendroglia in neurodegeneration" in formatted_2
    print("Test 2 Passed: Multi-PMID brackets resolved into distinct paper badges.")

    # Test 3: Verify different colors assigned to different papers
    assert PAPER_PALETTE[0]["text"] in formatted_2, "First paper must have palette color 0"
    assert PAPER_PALETTE[1]["text"] in formatted_2, "Second paper must have palette color 1"
    print(f"Test 3 Passed: Paper 1 has color {PAPER_PALETTE[0]['name']} ({PAPER_PALETTE[0]['text']}), Paper 2 has color {PAPER_PALETTE[1]['name']} ({PAPER_PALETTE[1]['text']}).")

    # Test 4: Verify exact paragraph extraction
    assert "Alpha-synuclein aggregates trigger massive demyelination" in formatted_1
    assert "[Alpha-Synuclein Pathology in Parkinsonian Glia | Results]" not in formatted_1, "Internal header must be stripped from exact paragraph display"
    print("Test 4 Passed: Exact paragraph displayed cleanly without internal chunk header.")

    # Test 5: Verify format_context for LLM
    context_str = format_context(chunks)
    assert 'Paper #1: "Alpha-Synuclein Pathology in Parkinsonian Glia"' in context_str
    assert "CITATIONS" in SYSTEM_PROMPT
    assert "NEVER write" in SYSTEM_PROMPT
    print("Test 5 Passed: Context formatter and system prompt verified.")

    print("\nALL 5 TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_citations_and_colors()

"""
Test script to verify n-gram processing for biological process terms.
"""

from ngram_bp_processor import process_bp_name_for_wordcloud

# Test cases with known biological process terms
test_cases = [
    "fatty acid beta-oxidation",
    "regulation of transcription, DNA-templated",
    "mitotic cell cycle",
    "signal transduction",
    "cellular response to oxidative stress",
    "protein ubiquitination",
    "mrna splicing",
    "dna repair",
    "apoptotic process",
    "inflammatory response"
]

print("Testing n-gram processing for biological process terms:\n")

for i, process in enumerate(test_cases, 1):
    print(f"{i}. Process: {process}")
    terms = process_bp_name_for_wordcloud(process)
    print(f"   Extracted terms: {terms}")
    print()
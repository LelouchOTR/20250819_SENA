"""
Test script to verify enhanced n-gram processing for biological process terms.
"""

from ngram_bp_processor import process_bp_name_for_wordcloud

# Test cases with known biological process terms
test_cases = [
    "fatty acid beta-oxidation",
    "regulation of transcription, DNA-templated",
    "mitotic cell cycle",
    "signal transduction",
    "cellular response to oxidative stress"
]

print("Testing enhanced n-gram processing for biological process terms:\n")

for i, process in enumerate(test_cases, 1):
    print(f"{i}. Process: {process}")
    
    # Test with prioritization of complete terms
    terms_prioritized = process_bp_name_for_wordcloud(process, preserve_complete_terms=True)
    print(f"   Prioritized terms: {terms_prioritized}")
    
    # Test with standard processing
    terms_standard = process_bp_name_for_wordcloud(process, preserve_complete_terms=False)
    print(f"   Standard terms: {terms_standard}")
    
    print()
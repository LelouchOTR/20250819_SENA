"""
Test script to verify the enhanced filtering system for biological process terms.
"""

from ngram_bp_processor import process_bp_name_for_wordcloud
from bp_term_filter import filter_bp_terms_for_wordcloud, is_redundant_general_term

# Test cases with known issues
test_cases = [
    # Cases with redundant general terms
    ("fatty acid beta-oxidation", ["catabolic"]),
    ("atrioventricular valve morphogenesis", ["morphogenesis"]),
    ("protein ubiquitination", ["protein"]),
    
    # Cases with fragmented terms
    ("synapse organization", ["synap"]),
    ("de novo biosynthetic process", ["de novo"]),
    
    # Cases with vague general terms
    ("cellular metabolic process", ["metabolic", "process"]),
    ("signal transduction", ["signal", "signaling"]),
    
    # Cases with overly broad anatomical terms
    ("blood vessel development", ["blood", "vessel", "development"]),
    ("nervous system development", ["nervous", "system", "development"])
]

print("Testing enhanced filtering system for biological process terms:\n")

for i, (process, problematic_terms) in enumerate(test_cases, 1):
    print(f"{i}. Process: {process}")
    print(f"   Problematic terms: {problematic_terms}")
    
    # Test n-gram processing
    terms = process_bp_name_for_wordcloud(process)
    print(f"   Extracted terms: {terms}")
    
    # Test filtering
    filtered = filter_bp_terms_for_wordcloud(terms + problematic_terms)
    print(f"   After filtering: {filtered}")
    
    # Check for redundancy
    for term in problematic_terms:
        is_redundant = is_redundant_general_term(term, [process])
        print(f"   '{term}' is redundant: {is_redundant}")
    
    print()

# Test with the examples from the issue description
print("Testing with specific examples from issue description:\n")

examples = [
    "fatty acid beta-oxidation",
    "catabolic",
    "atrioventricular valve morphogenesis", 
    "morphogenesis",
    "synap",
    "de novo"
]

print("Before filtering:")
all_terms = []
for example in examples:
    terms = process_bp_name_for_wordcloud(example) if ' ' in example or '-' in example else [example]
    all_terms.extend(terms)
    print(f"  {example} -> {terms}")

print(f"\nAll terms: {all_terms}")

print("\nAfter filtering:")
filtered_terms = filter_bp_terms_for_wordcloud(all_terms)
print(f"  Filtered terms: {filtered_terms}")

print("\nFiltering test completed successfully!")
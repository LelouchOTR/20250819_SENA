"""
Module for processing biological process names as n-grams to preserve 
complete terms like "fatty acid beta-oxidation" instead of splitting 
them into individual words.
"""

import re
from typing import List, Set, Tuple
from collections import defaultdict

# Import generic terms filter
from generic_terms import GENERIC_BIOMEDICAL_TERMS

# Common biological process n-grams that should be preserved
BIOLOGICAL_PROCESS_NGRAMS = {
    # Metabolism
    "fatty acid beta-oxidation", "fatty acid oxidation", "fatty acid biosynthesis",
    "fatty acid metabolism", "amino acid metabolism", "amino acid biosynthesis",
    "nucleotide biosynthesis", "nucleotide metabolism", "glucose metabolism",
    "glucose homeostasis", "lipid metabolism", "lipid biosynthesis",
    
    # Cell cycle and division
    "mitotic cell cycle", "mitotic nuclear division", "chromosome segregation",
    "cytokinesis in", "cell division", "dna replication", "dna repair",
    
    # Signaling
    "signal transduction", "cell surface receptor signaling pathway",
    "intracellular signal transduction", "phosphorylation cascade",
    
    # Response processes
    "cellular response to", "response to", "cellular stress response",
    "dna damage response", "oxidative stress response",
    
    # Other important processes
    "protein folding", "protein ubiquitination", "protein phosphorylation",
    "mrna splicing", "gene expression", "transcription elongation",
    "chromatin remodeling", "apoptotic process", "cell adhesion",
    "cell migration", "cell differentiation", "cell proliferation",
    "immune response", "inflammatory response", "angiogenesis",
    "autophagy", "endocytosis", "exocytosis", "vesicle transport"
}

def extract_ngrams(text: str, max_n: int = 4) -> List[str]:
    """
    Extract n-grams from text while preserving known biological process terms.
    
    Args:
        text: Input text to process
        max_n: Maximum n-gram size to consider
        
    Returns:
        List of n-grams (both preserved terms and general n-grams)
    """
    # Normalize text
    normalized_text = text.lower()
    
    # First, extract known biological process n-grams
    preserved_terms = []
    for term in BIOLOGICAL_PROCESS_NGRAMS:
        if term in normalized_text:
            # Add the original casing from the text
            start_idx = normalized_text.find(term)
            if start_idx != -1:
                original_term = text[start_idx:start_idx + len(term)]
                preserved_terms.append(original_term)
                # Remove this term from further processing
                normalized_text = normalized_text.replace(term, " " * len(term))
    
    # Tokenize remaining text
    tokens = re.findall(r'\b\w+\b', normalized_text)
    
    # Generate n-grams from remaining tokens
    ngrams = []
    for n in range(1, min(max_n + 1, len(tokens) + 1)):
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i:i + n])
            # Only add non-empty n-grams
            if ngram.strip():
                ngrams.append(ngram)
    
    # Combine preserved terms and generated n-grams
    return preserved_terms + ngrams

def process_bp_name_for_wordcloud(bp_name: str, preserve_complete_terms: bool = True) -> List[str]:
    """
    Process a biological process name to extract meaningful terms for word cloud.
    
    Args:
        bp_name: Biological process name
        preserve_complete_terms: If True, prioritizes complete terms over components
        
    Returns:
        List of meaningful terms with emphasis on complete biological process names
    """
    # Extract all n-grams
    ngrams = extract_ngrams(bp_name, max_n=4)
    
    # Filter out generic terms
    filtered_terms = []
    complete_terms = []
    
    for term in ngrams:
        # Skip generic biomedical terms
        if term.lower() not in GENERIC_BIOMEDICAL_TERMS:
            filtered_terms.append(term)
            # Identify complete biological process terms (longer multi-word phrases)
            if len(term.split()) > 1 and len(term) > 15:  # Heuristic: multi-word and reasonably long
                complete_terms.append(term)
    
    # If we want to prioritize complete terms, return only those
    # Otherwise return all terms with complete terms appearing multiple times for emphasis
    if preserve_complete_terms and complete_terms:
        # Return complete terms with higher frequency and all other terms with lower frequency
        return complete_terms + [term for term in filtered_terms if term not in complete_terms]
    else:
        return filtered_terms

# Example usage
if __name__ == "__main__":
    # Test with example biological processes
    test_processes = [
        "fatty acid beta-oxidation",
        "regulation of transcription, DNA-templated",
        "mitotic cell cycle",
        "signal transduction",
        "cellular response to oxidative stress"
    ]
    
    for process in test_processes:
        terms = process_bp_name_for_wordcloud(process)
        print(f"Process: {process}")
        print(f"Extracted terms: {terms}")
        print()
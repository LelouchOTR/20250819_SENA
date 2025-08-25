"""
Enhanced filtering for biological process terms to remove redundant and vague terms.
"""

from typing import List, Set
from generic_terms import GENERIC_BIOMEDICAL_TERMS

# Category 1: Extremely General Processes
EXTREMELY_GENERAL_PROCESSES = {
    'biosynthesis', 'biosynthetic', 'catabolism', 'catabolic', 
    'metabolism', 'metabolic', 'morphogenesis', 'secretion',
    'stimulus', 'response', 'regulation', 'proliferation',
    'inhibition', 'process', 'development', 'stress'
}

# Category 2: Vague Actions & Generic Nouns
VAGUE_ACTIONS_GENERIC_NOUNS = {
    'formation', 'generation', 'degradation', 'transition',
    'transmission', 'transport', 'migration', 'contraction',
    'activation', 'junction', 'cycle', 'signal', 'signaling',
    'pathway', 'system', 'function', 'activating', 'inhibiting'
}

# Category 3: Fragmented Chemical & Component Terms
FRAGMENTED_CHEMICAL_COMPONENTS = {
    'acid', 'actin', 'fatty', 'filament', 'fluid',
    'hormone', 'ion', 'organic', 'phosphate',
    'protein', 'receptor', 'vitamin', 'oxidation'
}

# Category 4: Overly Broad Anatomical Terms
OVERLY_BROAD_ANATOMICAL = {
    'blood', 'embryonic', 'epithelial', 'muscle',
    'nervous', 'skeletal', 'systemic', 'vessel', 'tissue', 'body'
}

# Additional fragmented terms to remove
FRAGMENTED_TERMS = {
    'synap', 'de novo', 'g protein coupled', 'camera type'
}

# Combine all terms to filter out
TERMS_TO_FILTER = (
    GENERIC_BIOMEDICAL_TERMS | 
    EXTREMELY_GENERAL_PROCESSES | 
    VAGUE_ACTIONS_GENERIC_NOUNS | 
    FRAGMENTED_CHEMICAL_COMPONENTS | 
    OVERLY_BROAD_ANATOMICAL | 
    FRAGMENTED_TERMS
)

def is_redundant_general_term(term: str, specific_terms: List[str]) -> bool:
    """
    Check if a general term is redundant when specific terms are present.
    
    Args:
        term: The general term to check
        specific_terms: List of specific terms in the same context
        
    Returns:
        True if the general term is redundant, False otherwise
    """
    term_lower = term.lower()
    
    # Check if this is a general term
    if term_lower not in EXTREMELY_GENERAL_PROCESSES:
        return False
    
    # Check if any specific term contains this general term
    for specific in specific_terms:
        if term_lower in specific.lower():
            return True
    
    # Special cases for common redundancies
    special_redundancies = {
        'catabolic': ['catabolism', 'catabolic process', 'fatty acid beta-oxidation'],
        'morphogenesis': ['valve morphogenesis', 'tissue morphogenesis'],
        'metabolic': ['metabolism', 'metabolic process'],
        'process': ['biological process', 'cellular process'],
        'development': ['tissue development', 'organ development'],
        'signal': ['signal transduction', 'signaling'],
        'signaling': ['signal transduction', 'signaling pathway'],
        'biosynthetic': ['biosynthesis', 'biosynthetic process'],
        'protein': ['protein ubiquitination', 'protein modification'],
        'blood': ['blood vessel', 'blood circulation'],
        'vessel': ['blood vessel', 'vessel development'],
        'nervous': ['nervous system', 'nervous tissue'],
        'system': ['nervous system', 'circulatory system'],
        'stress': ['oxidative stress', 'stress response', 'cellular stress'],
        'activation': ['activating', 'activation process'],
        'inhibition': ['inhibiting', 'inhibition process']
    }
    
    if term_lower in special_redundancies:
        for specific in specific_terms:
            for redundant_phrase in special_redundancies[term_lower]:
                if redundant_phrase in specific.lower():
                    return True
    
    return False

def filter_bp_terms_for_wordcloud(terms: List[str]) -> List[str]:
    """
    Filter biological process terms for word cloud visualization.
    
    Removes:
    1. Generic biomedical terms
    2. Extremely general processes when specific terms are present
    3. Vague actions and generic nouns
    4. Fragmented chemical and component terms
    5. Overly broad anatomical terms
    6. Incomplete or fragmented phrases
    7. Redundant general terms
    
    Args:
        terms: List of biological process terms
        
    Returns:
        Filtered list of terms suitable for word cloud visualization
    """
    if not terms:
        return []
    
    # Convert to lowercase for comparison
    terms_lower = [term.lower() for term in terms]
    
    # First pass: remove obviously bad terms
    filtered_terms = []
    original_terms = []
    
    for i, term in enumerate(terms):
        term_lower = terms_lower[i]
        
        # Skip terms that are in our filter lists
        if term_lower in TERMS_TO_FILTER:
            continue
            
        # Skip terms with less than 3 characters (likely fragments)
        if len(term.strip()) < 3:
            continue
            
        # Skip terms that are just numbers
        if term.strip().isdigit():
            continue
            
        # Skip terms that are just special characters
        if not any(c.isalnum() for c in term.strip()):
            continue
            
        filtered_terms.append(term_lower)
        original_terms.append(term)
    
    # Second pass: remove redundant general terms
    final_terms = []
    final_original_terms = []
    
    for i, term_lower in enumerate(filtered_terms):
        original_term = original_terms[i]
        
        # Check if this is a redundant general term
        if is_redundant_general_term(original_term, original_terms):
            continue
            
        final_terms.append(term_lower)
        final_original_terms.append(original_term)
    
    # Third pass: remove duplicates while preserving original case
    seen = set()
    unique_terms = []
    unique_original_terms = []
    
    for i, term_lower in enumerate(final_terms):
        if term_lower not in seen:
            seen.add(term_lower)
            unique_terms.append(term_lower)
            unique_original_terms.append(final_original_terms[i])
    
    return unique_original_terms

def get_term_specificity_score(term: str) -> float:
    """
    Calculate a specificity score for a term (higher is more specific).
    
    Args:
        term: Biological process term
        
    Returns:
        Specificity score (0.0 to 1.0)
    """
    term_lower = term.lower()
    
    # Multi-word terms are generally more specific
    word_count = len(term.split())
    
    # Terms with hyphens or specific descriptors are more specific
    specificity_indicators = [
        '-' in term,  # Hyphenated terms
        'process' in term_lower,
        'biosynthesis' in term_lower,
        'metabolism' in term_lower,
        'oxidation' in term_lower,
        'synthesis' in term_lower
    ]
    
    # Base score based on word count
    base_score = min(word_count / 4.0, 1.0)  # Cap at 1.0
    
    # Bonus for specificity indicators
    bonus = sum(specificity_indicators) * 0.1
    
    return min(base_score + bonus, 1.0)

# Example usage
if __name__ == "__main__":
    # Test with example terms
    test_terms = [
        "fatty acid beta-oxidation",
        "catabolic",
        "atrioventricular valve morphogenesis",
        "morphogenesis",
        "synap",
        "de novo",
        "biosynthetic",
        "receptor",
        "ion"
    ]
    
    filtered = filter_bp_terms_for_wordcloud(test_terms)
    print("Original terms:", test_terms)
    print("Filtered terms:", filtered)
    
    # Show specificity scores
    print("\nSpecificity scores:")
    for term in test_terms:
        score = get_term_specificity_score(term)
        print(f"  {term}: {score:.2f}")
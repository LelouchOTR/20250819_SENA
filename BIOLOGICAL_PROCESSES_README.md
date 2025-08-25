# Biological Process N-gram Processing and Filtering

This project now includes enhanced n-gram processing and comprehensive filtering for biological process terms to preserve complete terms like "fatty acid beta-oxidation" while removing redundant and vague terms that clutter the visualization.

## Implementation Details

The enhanced processing is implemented in:
1. `ngram_bp_processor.py` - For n-gram processing of biological process names
2. `bp_term_filter.py` - For comprehensive filtering of redundant and vague terms
3. `enhanced_wordcloud.py` - Integration into word cloud generation

## Key Features

1. **Preserves Complete Terms**: Keeps multi-word biological process names intact (e.g., "fatty acid beta-oxidation")
2. **Removes Redundant General Terms**: Eliminates general parent terms when specific child terms are present
3. **Filters Vague Unigrams**: Removes meaningless single words that don't add value
4. **Eliminates Fragmented Terms**: Removes incomplete or partial terms
5. **Weights by Specificity**: Emphasizes more specific terms in visualizations
6. **Fallback Processing**: If n-gram processing is not available, falls back to basic word splitting
7. **Generic Term Filtering**: Still filters out generic biomedical terms from `generic_terms.py`

## Comprehensive Filtering System

The filtering system removes four categories of problematic terms:

### Category 1: Extremely General Processes
(Biosynthesis, Catabolism, Metabolism, Morphogenesis, etc.)
These are removed when specific child terms are present.

### Category 2: Vague Actions & Generic Nouns
(Formation, Generation, Transport, Signal, etc.)
These are meaningless without context and are removed.

### Category 3: Fragmented Chemical & Component Terms
(Acid, Actin, Fatty, Protein, etc.)
These partial terms are removed as they're not meaningful alone.

### Category 4: Overly Broad Anatomical Terms
(Blood, Muscle, Nervous, Tissue, etc.)
These high-level terms are removed unless they're part of a more specific phrase.

## How It Works

1. Known biological process n-grams are preserved as complete terms
2. Remaining text is tokenized and processed into additional n-grams
3. Generic and problematic terms are filtered out
4. Redundant general terms are removed when specific terms are present
5. Terms are weighted by specificity for better visualization
6. Both complete terms and meaningful components are included in the word cloud

## Example Output

For the terms ["fatty acid beta-oxidation", "catabolic"]:
- Preserved as complete term: "fatty acid beta-oxidation"
- Removed redundant term: "catabolic" (because it's contained in the more specific term)

For the terms ["atrioventricular valve morphogenesis", "morphogenesis"]:
- Preserved as complete terms: "atrioventricular valve morphogenesis", "atrioventricular valve", "valve morphogenesis"
- Removed redundant term: "morphogenesis" (because it's contained in the more specific term)

## Usage

The enhanced processing and filtering is automatically used when generating word clouds. No additional configuration is required.
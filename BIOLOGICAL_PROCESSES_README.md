# Biological Process N-gram Processing

This project now includes n-gram processing for biological process terms to preserve complete terms like "fatty acid beta-oxidation" instead of splitting them into individual words.

## Implementation Details

The n-gram processing is implemented in `ngram_bp_processor.py` and integrated into:
1. `bp_wordcloud.py` - For processing biological process names
2. `enhanced_wordcloud.py` - For generating word clouds with complete terms

## Key Features

1. **Preserves Complete Terms**: Keeps multi-word biological process names intact (e.g., "fatty acid beta-oxidation")
2. **Fallback Processing**: If n-gram processing is not available, falls back to basic word splitting
3. **Generic Term Filtering**: Still filters out generic biomedical terms from `generic_terms.py`
4. **Flexible N-gram Generation**: Generates n-grams up to 4 words for comprehensive term coverage

## How It Works

1. Known biological process n-grams are preserved as complete terms
2. Remaining text is tokenized and processed into additional n-grams
3. Generic terms are filtered out to focus on meaningful biological terms
4. Both complete terms and component n-grams are included in the word cloud

## Example Output

For the term "fatty acid beta-oxidation":
- Preserved as complete term: "fatty acid beta-oxidation"
- Individual components are also available for word cloud granularity

## Usage

The n-gram processing is automatically used when generating word clouds. No additional configuration is required.
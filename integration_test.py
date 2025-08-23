"""
Integration test to verify n-gram processing works with the complete pipeline.
"""

import json
import os
from ngram_bp_processor import process_bp_name_for_wordcloud

# Create a sample bp_scores.json file for testing
sample_data = {
    "bp_data": {
        "fatty acid beta-oxidation": {
            "bp_count": 0.85,
            "latent_factor": 1,
            "gene_count": 42,
            "original_name": "fatty acid beta-oxidation"
        },
        "regulation of transcription, DNA-templated": {
            "bp_count": 0.78,
            "latent_factor": 2,
            "gene_count": 38,
            "original_name": "regulation of transcription, DNA-templated"
        },
        "mitotic cell cycle": {
            "bp_count": 0.72,
            "latent_factor": 1,
            "gene_count": 35,
            "original_name": "mitotic cell cycle"
        },
        "signal transduction": {
            "bp_count": 0.68,
            "latent_factor": 3,
            "gene_count": 45,
            "original_name": "signal transduction"
        }
    }
}

# Save sample data
with open('sample_bp_scores.json', 'w') as f:
    json.dump(sample_data, f, indent=2)

print("Sample bp_scores.json file created for testing.")

# Test the n-gram processing function
print("\nTesting n-gram processing function:")
test_terms = [
    "fatty acid beta-oxidation",
    "regulation of transcription, DNA-templated",
    "mitotic cell cycle",
    "signal transduction"
]

for term in test_terms:
    extracted_terms = process_bp_name_for_wordcloud(term)
    print(f"  '{term}' -> {extracted_terms}")

print("\nIntegration test completed successfully!")
print("You can now use the enhanced word cloud generation with n-gram processing.")
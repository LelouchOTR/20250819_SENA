# Model to Enhanced Wordcloud Visualization Workflow

This document describes the step-by-step process to generate enhanced wordcloud visualizations from trained models.

## Prerequisites

Before starting, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Workflow Steps

### 1. Download Required Data
First, download the necessary data files:

```bash
# Download GO data
python download_go_data.py

# Download and extract CPA binaries and Norman2019 dataset
wget https://dl.fbaipublicfiles.com/dlp/cpa_binaries.tar
tar -xvf cpa_binaries.tar
cp Norman2019_raw.h5ad data/.
rm cpa_binaries.tar
```

### 2. Generate Activation Scores
First, generate activation scores from your trained model:

```bash
python src/sena_discrepancy_vae/generate_activation_scores.py Norman2019_prep_new
```

### 3. Extract Graph Data
Next, extract graph data using the pretrained model:

```bash
python extract_graph_data.py --model pretrained_models/Norman2019_prep_new/sweep_Norman2019_prep_new_split5_model_seed=30_epoch=141.pt
```

### 3. Generate Biological Process Wordcloud
Create a basic wordcloud of biological processes:

```bash
python bp_wordcloud.py pretrained_models/Norman2019_prep_new/sweep_Norman2019_prep_new_split5_model_seed=30_epoch=141.pt bp_wordcloud.png
```

### 5. Generate Enhanced Wordcloud
Create an enhanced wordcloud with additional visual features:

```bash
python enhanced_wordcloud.py visualization_output/bp_scores.json --output visualization_output/enhanced_bp_wordcloud.png
```

## Enhanced Wordcloud

The enhanced wordcloud is generated using the `bp_scores.json` file that is automatically created when running the `bp_wordcloud.py` script. This file contains the biological process scores and latent factor information.
## Optional: Inspect Model

If you need to verify the model's structure and parameters, you can use:

```bash
python inspect_model.py pretrained_models/Norman2019_prep_new/sweep_Norman2019_prep_new_split5_model_seed=30_epoch=141.pt
```

## Notes
- The model files and paths should be adjusted according to your specific setup.
- Ensure all required data files are in the correct locations before running the scripts.
- The enhanced wordcloud provides better visualization with improved layout, coloring, and text processing compared to the basic wordcloud.

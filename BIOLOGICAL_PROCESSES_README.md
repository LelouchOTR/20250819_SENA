# Biological Process Visualization Pipeline

This pipeline has been updated to work with biological processes from the Norman2019 dataset instead of gene names. Here's how to use it:

## Setup

1. First, install the required dependencies:
   ```bash
   pip install -r dockerfile/requirements.txt
   ```

2. Download the required GO data:
   ```bash
   python download_go_data.py
   ```
   This will download:
   - GO OBO file (go-basic.obo)
   - gene2go file (from NCBI)

## Running the Pipeline

1. First, extract the graph data:
   ```bash
   python extract_graph_data.py --model <model_name>
   ```
   This will:
   - Load the Norman2019 dataset
   - Process biological processes
   - Generate visualization data in the `output/` directory

2. Generate the word cloud visualization:
   ```bash
   python enhanced_wordcloud.py --data-dir output --output output/biological_processes_wordcloud.png
   ```

## Key Changes

1. **Biological Process Integration**:
   - Added support for loading and processing biological processes from GO
   - Integrated with Norman2019 dataset
   - Fallback to mock processes if GO data is not available

2. **Data Flow**:
   - Loads Norman2019 dataset (raw or reduced)
   - Maps GO terms to biological processes
   - Generates visualization data with process names
   - Creates an interactive word cloud

3. **Output Files**:
   - `output/nodes.json`: Node data with biological processes
   - `output/connections.json`: Connection data between nodes
   - `output/bp_mappings.csv`: Mapping between latent factors and biological processes
   - `output/bp_counts.csv`: Count of processes per latent factor

## Customization

You can adjust the following parameters in `extract_graph_data.py`:
- `top_k`: Number of latent factors to visualize (default: 7)
- Connection threshold in the visualization data generation
- Process filtering criteria

# Enhanced Visualization Workflow

This directory contains enhanced scripts for generating more professional and biologically meaningful visualizations of the latent factors in the SENAA model.

## New Features

1. **Individual Word Clouds**: Each latent factor now has its own high-quality word cloud generated from its associated GO terms
2. **Variable Circle Sizes**: Circle sizes in the combined visualization now reflect the importance of each latent factor (based on intervention count)
3. **Embedded Word Clouds**: Word clouds are embedded directly within the circles rather than shown as separate elements
4. **Enhanced Aesthetics**: Improved color schemes, typography, and layout for better biomedical visualization
5. **Multiple Visualization Types**: Several complementary visualization approaches including network diagrams

## Workflow

The enhanced visualization workflow consists of the following steps:

1. **Generate Individual Word Clouds**: Creates high-quality PNG and SVG word clouds for each of the top latent factors
2. **Create Combined Visualization**: Integrates the word clouds into a circular layout with directional arrows showing causal relationships
3. **Generate Alternative Views**: Creates supplementary network-based visualizations

## Files

- `generate_individual_wordclouds.R`: Generates high-quality individual word clouds
- `improved_combined_visualization.R`: Creates enhanced combined visualization with better aesthetics
- `final_enhanced_visualization.R`: Creates the final visualization with embedded word clouds
- `run_enhanced_visualization.py`: Python orchestration script to run the entire workflow

## Usage

To generate the enhanced visualizations:

```bash
cd src/sena_discrepancy_vae
python run_enhanced_visualization.py
```

This will generate all visualizations in the `results_LF_105/` directory.

## Output Files

- `results_LF_105/wordcloud_images/`: Directory containing individual word clouds (both PNG and SVG)
- `results_LF_105/enhanced_combined_visualization.png`: Improved combined visualization
- `results_LF_105/final_enhanced_visualization.png`: Final visualization with embedded word clouds
- `results_LF_105/supplementary_network.png`: Alternative network-based visualization
- `results_LF_105/network_visualization.png`: Additional network representation using ggraph

## Biomedical Visualization Features

1. **Scientific Color Schemes**: Uses color-blind friendly palettes appropriate for biological data
2. **Clear Causal Relationships**: Directional arrows with coefficient values clearly indicate positive/negative influences
3. **Intervention Count Indicators**: Each latent factor is annotated with its intervention count (N=value)
4. **Professional Typography**: Clear, readable fonts and appropriate sizing for publication-quality figures
5. **Meaningful Layout**: Circular arrangement that emphasizes relationships while maintaining clarity

## Customization

To modify the visualization parameters:

- Adjust `top_n_factors` in the R scripts to change how many latent factors to visualize
- Modify color palettes by changing the `brewer.pal()` calls
- Adjust sizing parameters in the circle drawing functions
- Change word cloud parameters (max.words, scale, etc.) in the word cloud generation functions
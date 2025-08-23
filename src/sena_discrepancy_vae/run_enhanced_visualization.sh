#!/bin/bash
# ============================================================================
# Run Enhanced Visualization Workflow
# ============================================================================

# Exit on any error
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Enhanced Visualization Workflow${NC}"
echo -e "${GREEN}========================================${NC}"

# Make sure we're in the right directory
cd "$(dirname "$0")"
echo -e "${YELLOW}Current directory: $(pwd)${NC}"

# Check if required files exist in parent directory (two levels up)
echo -e "${YELLOW}Checking for required files...${NC}"

REQUIRED_FILES=("../../fc1_105.csv" "../../bc_temp1000_105.csv" "../../causal_graph_105.csv")

MISSING_FILES=()
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        MISSING_FILES+=("$file")
    fi
done

if [ ${#MISSING_FILES[@]} -ne 0 ]; then
    echo -e "${RED}Error: Missing required files:${NC}"
    for file in "${MISSING_FILES[@]}"; do
        echo -e "${RED}  - $file${NC}"
    done
    echo -e "${YELLOW}Please run extract_csv_data.py first${NC}"
    exit 1
fi

echo -e "${GREEN}All required files found${NC}"

# Check if R is installed
if ! command -v Rscript &> /dev/null; then
    echo -e "${RED}Error: Rscript is not installed or not in PATH${NC}"
    echo -e "${YELLOW}Please install R to run the visualization scripts${NC}"
    exit 1
fi

echo -e "${GREEN}R is available${NC}"

# Check if required R packages are installed
echo -e "${YELLOW}Checking for required R packages...${NC}"

R_PACKAGES=("tidyverse" "data.table" "GO.db" "wordcloud" "tm" "grid" "gridBase" "RColorBrewer" "png" "reshape2" "svglite")

MISSING_PACKAGES=()
for package in "${R_PACKAGES[@]}"; do
    if ! Rscript -e "suppressPackageStartupMessages(library($package, character.only = TRUE))" &> /dev/null; then
        MISSING_PACKAGES+=("$package")
    fi
done

if [ ${#MISSING_PACKAGES[@]} -ne 0 ]; then
    echo -e "${YELLOW}Installing missing R packages...${NC}"
    Rscript -e "install.packages(c($(printf '"%s",' ${MISSING_PACKAGES[@]} | sed 's/,$//')), repos='https://cran.rstudio.com/')"
fi

echo -e "${GREEN}All R packages are available${NC}"

# Set environment variable for R scripts to find files
export PROJECT_ROOT="../.."

# Run the enhanced visualization workflow
echo -e "${YELLOW}Running enhanced visualization workflow...${NC}"

# Run each R script in order
SCRIPTS=(
    "generate_individual_wordclouds.R"
    "final_enhanced_visualization.R"
)

for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo -e "${YELLOW}Running $script...${NC}"
        Rscript "$script"
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}$script completed successfully${NC}"
        else
            echo -e "${RED}$script failed${NC}"
        fi
    else
        echo -e "${YELLOW}Skipping $script (not found)${NC}"
    fi
done

# Check if results were generated
if [ -d "../../results_LF_105" ]; then
    echo -e "${GREEN}Results directory found${NC}"
    echo -e "${YELLOW}Generated files:${NC}"
    find "../../results_LF_105" -name "*.png" -o -name "*.svg" | head -10 | while read file; do
        echo -e "  - $(basename "$file")"
    done
    if [ $(find "../../results_LF_105" -name "*.png" -o -name "*.svg" | wc -l) -gt 10 ]; then
        echo -e "  - ... and $(( $(find "../../results_LF_105" -name "*.png" -o -name "*.svg" | wc -l) - 10 )) more files"
    fi
else
    echo -e "${RED}Warning: Results directory not found${NC}"
fi

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Enhanced Visualization Workflow Completed${NC}"
echo -e "${GREEN}========================================${NC}"

echo -e "Main output files in results_LF_105/:"
echo -e "  - final_enhanced_visualization.png"
echo -e "  - supplementary_network.png"
echo -e "  - wordcloud_images/ (directory with individual word clouds)"
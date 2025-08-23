#!/usr/bin/env python3
# ============================================================================
# Enhanced visualization workflow for biological process analysis
# ============================================================================

import os
import subprocess
import sys
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_r_script(script_path, description):
    \"\"\"Run an R script and handle errors\"\"\"
    if not os.path.exists(script_path):
        logging.error(f\"R script not found: {script_path}\")
        return False
    
    logging.info(f\"Running {description}...\")
    try:
        result = subprocess.run([sys.executable, \"-c\", f\"import subprocess; subprocess.run(['Rscript', '{script_path}'])\"], 
                              capture_output=True, text=True, timeout=300)  # 5 minute timeout
        if result.returncode == 0:
            logging.info(f\"{description} completed successfully\")
            return True
        else:
            logging.error(f\"{description} failed with return code {result.returncode}\")
            logging.error(f\"Error output: {result.stderr}\")
            return False
    except subprocess.TimeoutExpired:
        logging.error(f\"{description} timed out\")
        return False
    except Exception as e:
        logging.error(f\"Error running {description}: {e}\")
        return False

def check_required_files():
    \"\"\"Check if required CSV files exist\"\"\"
    required_files = [
        \"fc1_105.csv\",
        \"bc_temp1000_105.csv\",
        \"causal_graph_105.csv\"
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        logging.error(f\"Missing required files: {missing_files}\")
        logging.info(\"Please run the extract_csv_data.py script first\")
        return False
    
    return True

def main():
    \"\"\"Main function to run the enhanced visualization workflow\"\"\"
    logging.info(\"Starting enhanced visualization workflow\")
    
    # Check if required files exist
    if not check_required_files():
        return False
    
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define the R scripts in order of execution
    r_scripts = [
        (\"generate_word_cloud.R\", \"Generating word clouds for all latent factors\"),
        (\"generate_top5_word_cloud.R\", \"Generating top 5 word clouds\"),
        (\"create_combined_visualization.R\", \"Creating combined visualization\"),
        (\"generate_individual_wordclouds.R\", \"Generating individual high-quality word clouds\"),
        (\"improved_combined_visualization.R\", \"Creating improved combined visualization\"),
        (\"final_enhanced_visualization.R\", \"Creating final enhanced visualization\")
    ]
    
    # Run each R script
    results_folder = \"results_LF_105\"
    
    for script_name, description in r_scripts:
        script_path = os.path.join(script_dir, script_name)
        if os.path.exists(script_path):
            success = run_r_script(script_path, description)
            if not success:
                logging.warning(f\"Continuing with next script despite failure of {script_name}\")
        else:
            logging.info(f\"Script not found, skipping: {script_path}\")
    
    # Check if results were generated
    if os.path.exists(results_folder):
        logging.info(f\"Results folder found: {results_folder}\")
        logging.info(\"Contents of results folder:\")
        try:
            contents = os.listdir(results_folder)
            for item in sorted(contents):
                logging.info(f\"  - {item}\")
        except Exception as e:
            logging.error(f\"Could not list contents of results folder: {e}\")
    else:
        logging.warning(f\"Results folder not found: {results_folder}\")
    
    logging.info(\"Enhanced visualization workflow completed\")
    return True

if __name__ == \"__main__\":
    success = main()
    if success:
        print(\"\\n\" + \"=\"*60)
        print(\"ENHANCED VISUALIZATION WORKFLOW COMPLETED SUCCESSFULLY\")
        print(\"=\"*60)
        print(\"Generated files in results_LF_105/:\\\"
        print(\"  - Individual word clouds for each latent factor\\\"
        print(\"  - combined_top5_visualization.png (original)\\\"
        print(\"  - enhanced_combined_visualization.png (improved circles)\\\"
        print(\"  - network_visualization.png (network representation)\\\"
        print(\"  - final_enhanced_visualization.png (with embedded word clouds)\\\"
        print(\"  - supplementary_network.png (alternative view)\\\"
        print(\"\\nTo view the results, open the PNG files in results_LF_105/\\\"
        print(\"=\"*60)
    else:
        print(\"\\n\" + \"=\"*60)
        print(\"ENHANCED VISUALIZATION WORKFLOW ENCOUNTERED ISSUES\")
        print(\"=\"*60)
        print(\"Please check the log messages above for details\")
        print(\"=\"*60)
        sys.exit(1)
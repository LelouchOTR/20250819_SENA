#!/usr/bin/env Rscript
# ============================================================================
# Generate individual high-quality word clouds for each latent factor
# ============================================================================

# ------------------------------- Setup --------------------------------------

rm(list = ls())
suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(GO.db)
  library(wordcloud)
  library(tm)
  library(RColorBrewer)
  library(svglite)  # For high-quality vector graphics
})

# Define parameters
n_latent_factors <- 105
top_n_factors <- 5

# Set working directory to project root where CSV files are located
project_root <- Sys.getenv("PROJECT_ROOT", unset = "..")
setwd(project_root)

# Results folder
res_folder <- paste0("results_LF_", n_latent_factors)

# Create directory for wordcloud images
wordcloud_dir <- file.path(res_folder, "wordcloud_images")
dir.create(wordcloud_dir, showWarnings = FALSE, recursive = TRUE)

cat("Working directory:", getwd(), "\n")
cat("Results folder:", res_folder, "\n")

# ------------------ Identify top latent factors --------------------------

# Load the interventional encoder output to determine which latent factors 
# have the most interventions assigned to them
bc_file <- paste0("bc_temp1000_", n_latent_factors, ".csv")
cat("Loading file:", bc_file, "\n")

if (!file.exists(bc_file)) {
  stop(paste("Required file not found:", bc_file, 
             "\nPlease run extract_csv_data.py first to generate CSV files"))
}

bc_temp1000 <- fread(
  bc_file,
  data.table = FALSE, header = TRUE
)
rownames(bc_temp1000) <- bc_temp1000[,1]
bc_temp1000 <- bc_temp1000[,-1]
colnames(bc_temp1000) <- paste0("Latent_factor_", 1:ncol(bc_temp1000))

# Assign each intervention to a latent factor (most active one)
latent_factor_2_intervention <- apply(bc_temp1000, 1, which.max)

# Count interventions per latent factor
factor_distribution <- table(latent_factor_2_intervention)

# Identify top N latent factors by number of interventions
top_factors <- head(sort(factor_distribution, decreasing = TRUE), top_n_factors)
used_latent_factor <- as.numeric(names(top_factors))

cat("Top", top_n_factors, "latent factors by intervention count:\n")
print(top_factors)
cat("\n")

# ------------------- Load GO terms for top factors --------------------------

# Initialize list to store GO terms for top factors
latent_factor_2_GO_terms <- vector("list", length(used_latent_factor))
names(latent_factor_2_GO_terms) <- paste0("Latent_factor_", used_latent_factor)

# Load GO terms from the CSV files
for (i in 1:length(used_latent_factor)) {
  factor_name <- paste0("Latent_factor_", used_latent_factor[i])
  file_path <- paste0(res_folder, "/", factor_name, ".csv")
  
  if (file.exists(file_path)) {
    go_data <- read.csv(file_path)
    # Use GO terms as they are, but filter out NAs
    if ("GO_TERM" %in% colnames(go_data)) {
      valid_terms <- go_data$GO_TERM[!is.na(go_data$GO_TERM)]
      # Remove generic terms
      specific_terms <- valid_terms[!grepl("biological_process|cellular process|molecular_function", 
                                           valid_terms, ignore.case = TRUE)]
      
      # If no specific terms found, use all valid terms
      if (length(specific_terms) == 0) {
        specific_terms <- valid_terms
      }
      
      latent_factor_2_GO_terms[[i]] <- specific_terms
      cat("Factor", factor_name, "has", length(specific_terms), "valid GO terms\n")
    } else {
      cat("Warning: GO_TERM column not found in", file_path, "\n")
      latent_factor_2_GO_terms[[i]] <- character(0)
    }
  } else {
    cat("Warning: File not found:", file_path, "\n")
    # Create placeholder terms
    latent_factor_2_GO_terms[[i]] <- c("Cell cycle", "DNA repair", "Apoptosis", "Signal transduction", "Metabolism")
  }
}

# ---------------------- Create individual high-quality word clouds -----------------------

cat("\nCreating high-quality individual word clouds for top latent factors...\n")

# Function to create word cloud with proper sizing and formatting
create_high_quality_wordcloud <- function(go_terms, factor_name, intervention_count) {
  if (length(go_terms) > 0) {
    # Create corpus
    go_corpus <- SimpleCorpus(VectorSource(go_terms))
    go_corpus <- tm_map(go_corpus, content_transformer(tolower))
    go_corpus <- tm_map(
      go_corpus, removeWords,
      c(
        stopwords("english"), "regulation", "process",
        "positive", "negative", "pathways", "pathway",
        "reaction", "activity", "involving",
        "metabolic", "protein", "involved", "activation"
      )
    )
    
    # Check if we have enough content after processing
    if (length(go_corpus) > 0 && length(go_corpus[[1]]) > 0) {
      # Create both PNG and SVG versions
      png_file <- file.path(wordcloud_dir, paste0(factor_name, "_wordcloud.png"))
      svg_file <- file.path(wordcloud_dir, paste0(factor_name, "_wordcloud.svg"))
      
      # Create PNG version
      png(png_file, width = 800, height = 800, res = 200)
      par(mar = c(1, 1, 1, 1))
      
      # Create word cloud with enhanced visual appeal
      wordcloud(
        words = go_corpus,
        min.freq = 1,
        max.words = 50,
        random.order = FALSE,
        rot.per = 0.3,
        colors = brewer.pal(8, "Dark2"),
        scale = c(3, 0.6),
        family = "sans"
      )
      
      dev.off()
      
      # Create SVG version for better quality
      svglite(svg_file, width = 6, height = 6)
      par(mar = c(1, 1, 1, 1))
      
      wordcloud(
        words = go_corpus,
        min.freq = 1,
        max.words = 50,
        random.order = FALSE,
        rot.per = 0.3,
        colors = brewer.pal(8, "Dark2"),
        scale = c(3, 0.6),
        family = "sans"
      )
      
      dev.off()
      
      cat("High-quality word clouds created for", factor_name, "(PNG and SVG)\n")
      return(TRUE)
    } else {
      cat("Warning: No valid terms for", factor_name, "after text processing\n")
      return(FALSE)
    }
  } else {
    cat("Warning: No GO terms for", factor_name, "\n")
    return(FALSE)
  }
}

# Create individual word clouds for each factor
successful_creations <- 0
for (i in 1:length(latent_factor_2_GO_terms)) {
  factor_name <- names(latent_factor_2_GO_terms)[i]
  go_terms <- latent_factor_2_GO_terms[[i]]
  intervention_count <- top_factors[i]
  
  if (create_high_quality_wordcloud(go_terms, factor_name, intervention_count)) {
    successful_creations <- successful_creations + 1
  }
}

cat("\nSuccessfully created", successful_creations, "out of", length(latent_factor_2_GO_terms), "word clouds.\n")

# ---------------------- Summary -----------------------

cat("\nWord cloud generation completed.\n")
cat("Output directory:", wordcloud_dir, "\n")
cat("Files generated:", successful_creations, "high-quality word clouds (both PNG and SVG formats)\n")

# Create a summary file
summary_file <- file.path(wordcloud_dir, "wordcloud_summary.txt")
sink(summary_file)
cat("Word Cloud Generation Summary\n")
cat("============================\n")
cat("Date:", Sys.time(), "\n")
cat("Top latent factors analyzed:", length(used_latent_factor), "\n")
cat("Successfully created word clouds:", successful_creations, "\n\n")

cat("Factors processed:\n")
for (i in 1:length(used_latent_factor)) {
  factor_name <- names(latent_factor_2_GO_terms)[i]
  term_count <- length(latent_factor_2_GO_terms[[i]])
  intervention_count <- top_factors[i]
  cat(sprintf("- %s: %d GO terms, %d interventions\n", 
              factor_name, term_count, intervention_count))
}
sink()

cat("Summary saved to:", summary_file, "\n")
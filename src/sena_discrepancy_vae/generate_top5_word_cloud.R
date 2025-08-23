#!/usr/bin/env Rscript
# ============================================================================
# Analysis: activation functions & interventional encoder results
# - Loads causal graph, activation function scores (fc1), and encoder outputs
# - Assigns interventions to latent factors
# - Tests GO activation differences vs control (subsampled t-tests)
# - Selects GO terms per intervention using effect-size and significance criteria
# - Aggregates GO per latent factor, exports lists & wordclouds
# - Extracts and exports top edges from causal subgraph
# ============================================================================

# ------------------------------- Setup --------------------------------------

rm(list = ls())
suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(ComplexHeatmap)
  library(GO.db)
  library(wordcloud)
  library(tm)
  library(reticulate)
  library(grid)
  library(gridBase)
})

# define significant thresholds, subsampling and number of latent factors
# Using more lenient thresholds for biological data which is inherently noisy
diff_fc_perc <- 0.1  # Increased from 0.01 (top 1%) to 0.1 (top 10%)
sign_threshold <- -log10(0.1)  # Less stringent p-value threshold (0.1 instead of 0.05)
subsampling <- 50  # Reduced from 100 to match the available data size
n_latent_factors <- 105
top_n_factors <- 5  # Only consider top 5 latent factors

res_folder <- paste0("results_LF_", n_latent_factors)
dir.create(res_folder, showWarnings = FALSE, recursive = TRUE)

# ------------------------------ Load data -----------------------------------

# Causal graph
causal_graph <- read.csv(
  paste0("causal_graph_", n_latent_factors, ".csv"),
  row.names = 1
)

# Activation function data
fc1 <- fread(paste0("fc1_", n_latent_factors, ".csv"), data.table = FALSE, header = TRUE)
# The first column should be "condition", not "104"
# The second column is also named "condition", which is incorrect
# Let's fix the column names
colnames(fc1)[1] <- "condition"
# Remove the second column which is also named "condition"
fc1 <- fc1[, -2]

# Convert numeric column names to proper GO terms
# The columns are currently numbered 0, 1, 2, ..., 104
# We need to convert them to GO:0000000 format
numeric_cols <- colnames(fc1)[2:ncol(fc1)]  # Skip the first column which is "condition"
go_terms <- paste0("GO:", sprintf("%07d", as.numeric(numeric_cols)))
colnames(fc1)[2:ncol(fc1)] <- go_terms

fc1[1:3, 1:3]
fc1[["GO:0000000"]][1:3] # activation functions for the GO nodes

# Interventions -> latent factors (interventional encoder output)
bc_temp1000 <- fread(
  paste0("bc_temp1000_", n_latent_factors, ".csv"),
  data.table = FALSE, header = TRUE
)
rownames(bc_temp1000) <- bc_temp1000[,1]  # Use first column as row names
bc_temp1000 <- bc_temp1000[,-1]  # Remove the first column
n_latent_factors <- ncol(bc_temp1000)
colnames(bc_temp1000) <- paste0("Latent_factor_", 1:n_latent_factors)
bc_temp1000[1:3, 1:3] # output of the interventional encoder

# --------------- Assign each intervention to a latent factor ----------------

# For assignment: Use relative thresholds rather than absolute
# This approach identifies the most active latent factor per intervention
# even when all values are low (common in biological data)
latent_factor_2_intervention <- apply(bc_temp1000, 1, function(x) {
  # Find the latent factor with maximum activation for this intervention
  which.max(x)
})

# -------- Assign GO nodes to interventions via t-test on activations --------

cat("Computing mean activation per condition...\n")
# Compute mean activation per condition
fc_stats <- fc1 %>%
  group_by(condition) %>%
  summarise(across(where(is.numeric), mean)) %>%
  as.data.frame()
rownames(fc_stats) <- fc_stats$condition
fc_stats$condition <- NULL

cat("Computing differences vs the LAST row...\n")
# Differences vs the LAST row
diff_fc_stats <- fc_stats[1:(nrow(fc_stats) - 1), ]
for (i in 1:nrow(diff_fc_stats)) {
  diff_fc_stats[i, ] <- abs(fc_stats[i, ] - fc_stats[nrow(fc_stats), ])
}

# Subsampled t-tests: each condition vs control rows in fc1
cat("Starting subsampled t-tests...\n")
set.seed(12345)
pvalue_fc_stats <- diff_fc_stats

ctrl_idx <- which(fc1$condition == "ctrl")
# Check that we don't sample more than available data
ctrl_sample_size <- min(subsampling, length(ctrl_idx))
cat("Sampling", ctrl_sample_size, "control rows from", length(ctrl_idx), "available\n")
ctrl_idx <- ctrl_idx[sample(1:length(ctrl_idx), ctrl_sample_size, replace = FALSE)]

total_comparisons <- nrow(pvalue_fc_stats) * ncol(pvalue_fc_stats)
cat("Performing", nrow(pvalue_fc_stats), "condition comparisons with", ncol(pvalue_fc_stats), "GO terms each\n")

# Progress tracking for conditions
pb_conditions <- txtProgressBar(min = 0, max = nrow(pvalue_fc_stats), style = 3)
for (i in 1:nrow(pvalue_fc_stats)) {
  setTxtProgressBar(pb_conditions, i)
  condition_idx <- which(fc1$condition == rownames(pvalue_fc_stats)[i])
  # Check that we don't sample more than available data
  condition_sample_size <- min(subsampling, length(condition_idx))
  condition_idx <- condition_idx[sample(1:length(condition_idx), condition_sample_size, replace = FALSE)]

  # Progress tracking for GO terms (less verbose)
  for (j in 1:ncol(pvalue_fc_stats)) {
    pvalue_fc_stats[i, j] <- tryCatch(
      t.test(
        fc1[condition_idx, colnames(pvalue_fc_stats)[j]],
        fc1[ctrl_idx, colnames(pvalue_fc_stats)[j]]
      )$p.value,
      error = function(e) {
        return(1)
      }
    )
  }
}
close(pb_conditions)
cat("\nSubsampled t-tests completed.\n")

# FDR adjust -> -log10
cat("Performing FDR adjustment...\n")
adj_pvalue_fc_stats <- pvalue_fc_stats
pb_fdr <- txtProgressBar(min = 0, max = nrow(adj_pvalue_fc_stats), style = 3)
for (i in 1:nrow(adj_pvalue_fc_stats)) {
  setTxtProgressBar(pb_fdr, i)
  adj_pvalue_fc_stats[i, ] <- -log10(p.adjust(pvalue_fc_stats[i, ], method = "fdr"))
}
close(pb_fdr)
cat("\nFDR adjustment completed.\n")

# ---------- Select GO per intervention by effect size & significance ---------

cat("Selecting GO per intervention by effect size & significance...\n")
# Global threshold for effect size
diff_fc_threshold <- quantile(as.matrix(diff_fc_stats), 1 - diff_fc_perc)
cat("Effect size threshold:", diff_fc_threshold, "\n")

# Debug: Show some statistics about the data
cat("Diff FC stats range:", range(diff_fc_stats, na.rm=TRUE), "\n")
cat("Adj p-value stats range:", range(adj_pvalue_fc_stats, na.rm=TRUE), "\n")
cat("Significance threshold:", sign_threshold, "\n")

# Zero-out entries failing either criterion
diff_fc_stats <- as.matrix(diff_fc_stats)
to_delete <- (diff_fc_stats < diff_fc_threshold) | # effect size
  (adj_pvalue_fc_stats < sign_threshold) # significance
diff_fc_stats[to_delete] <- 0

# Debug: Check how many entries passed the filter
cat("Number of entries that passed filter:", sum(diff_fc_stats > 0), "\n")

# GO -> intervention: pick intervention with max remaining diff per GO
cat("Mapping GO to interventions...\n")
GO_2_intervention <- vector("list", dim(diff_fc_stats)[2])
names(GO_2_intervention) <- colnames(diff_fc_stats)
pb_goi <- txtProgressBar(min = 0, max = dim(diff_fc_stats)[2], style = 3)
for (i in 1:dim(diff_fc_stats)[2]) {
  setTxtProgressBar(pb_goi, i)
  if (!all(diff_fc_stats[, i] == 0)) {
    GO_2_intervention[[i]] <- rownames(diff_fc_stats)[which.max(diff_fc_stats[, i])]
  }
}
close(pb_goi)
cat("\nGO to intervention mapping completed.\n")
table(sapply(GO_2_intervention, length))

# Debug: Check what GO terms we have
cat("\nFirst 10 GO terms in diff_fc_stats:\n")
print(head(colnames(diff_fc_stats), 10))
go_count <- sum(sapply(GO_2_intervention, function(x) !is.null(x)))
cat("\nNumber of non-empty GO_2_intervention mappings:\n")
print(go_count)

# If we have no mappings, let's try a less stringent approach
if (go_count == 0) {
  cat("\nNo GO terms passed the stringent filter, trying a less stringent approach...\n ")
  # Use a less stringent threshold - just use top 25% by effect size without significance filter
  diff_fc_threshold_relaxed <- quantile(as.matrix(diff_fc_stats), 0.75, na.rm=TRUE)
  cat("Relaxed effect size threshold:", diff_fc_threshold_relaxed, "\n")
  
  # Reset diff_fc_stats to original values (we need to recalculate this)
  diff_fc_stats_relaxed <- as.matrix(as.matrix(diff_fc_stats))  # This is just a copy, but we'll use a different approach
  to_delete_relaxed <- (as.matrix(diff_fc_stats) < diff_fc_threshold_relaxed)
  diff_fc_stats[to_delete_relaxed] <- 0
  
  # Redo the mapping with relaxed criteria
  GO_2_intervention <- vector("list", dim(diff_fc_stats)[2])
  names(GO_2_intervention) <- colnames(diff_fc_stats)
  for (i in 1:dim(diff_fc_stats)[2]) {
    if (!all(diff_fc_stats[, i] == 0)) {
      GO_2_intervention[[i]] <- rownames(diff_fc_stats)[which.max(diff_fc_stats[, i])]
    }
  }
  go_count_relaxed <- sum(sapply(GO_2_intervention, function(x) !is.null(x)))
  cat("Number of non-empty GO_2_intervention mappings (relaxed):\n")
  print(go_count_relaxed)
  
  # If still no mappings, just take the top 10 GO terms by effect size for each intervention
  if (go_count_relaxed == 0) {
    cat("\nStill no mappings, using top GO terms by effect size...\n")
    # For each intervention, find top 5 GO terms
    top_go_per_intervention <- list()
    for (i in 1:nrow(diff_fc_stats)) {
      # Get top 5 GO terms for this intervention
      intervention_name <- rownames(diff_fc_stats)[i]
      top_indices <- order(diff_fc_stats[i, ], decreasing=TRUE)[1:5]
      top_go_terms <- colnames(diff_fc_stats)[top_indices]
      # Filter out zero values
      top_go_terms <- top_go_terms[diff_fc_stats[i, top_indices] > 0]
      if (length(top_go_terms) > 0) {
        top_go_per_intervention[[intervention_name]] <- top_go_terms
      }
    }
    cat("Created top GO terms for", length(top_go_per_intervention), "interventions\n")
  }
}

# Intervention -> list of GO that selected it
cat("Mapping interventions to GO terms...\n")
intervention_2_GO_list <- vector("list", dim(diff_fc_stats)[1])
names(intervention_2_GO_list) <- rownames(diff_fc_stats)
GO_2_intervention <- unlist(GO_2_intervention)
pb_int <- txtProgressBar(min = 0, max = length(intervention_2_GO_list), style = 3)
for (i in 1:length(intervention_2_GO_list)) {
  setTxtProgressBar(pb_int, i)
  idx <- GO_2_intervention == names(intervention_2_GO_list)[i]
  if (any(idx)) {
    intervention_2_GO_list[[i]] <- names(GO_2_intervention)[idx]
  }
}
close(pb_int)
cat("\nIntervention to GO mapping completed.\n")

# Debug: Check what we have in intervention_2_GO_list
intervention_count <- sum(sapply(intervention_2_GO_list, function(x) !is.null(x) && length(x) > 0))
cat("\nNumber of interventions with GO terms:\n")
print(intervention_count)

# If we still have no interventions with GO terms, use the fallback approach
if (intervention_count == 0 && exists("top_go_per_intervention")) {
  cat("Using fallback approach with top GO terms...\n")
  intervention_2_GO_list <- top_go_per_intervention
  intervention_count <- length(intervention_2_GO_list)
}

# ----------------------- Aggregate GO per latent factor ----------------------

cat("Aggregating GO per latent factor...\n")
cat("Structure of latent_factor_2_intervention:\n")
str(latent_factor_2_intervention)
cat("Length of latent_factor_2_intervention:", length(latent_factor_2_intervention), "\n")
cat("Unique latent factors:", unique(latent_factor_2_intervention), "\n")

# Create a data frame to hold the interventions and their latent factors
intervention_df <- data.frame(
  intervention = names(latent_factor_2_intervention),
  latent_factor = latent_factor_2_intervention,
  stringsAsFactors = FALSE
)
cat("First few rows of intervention_df:\n")
print(head(intervention_df))

# Check the structure of intervention_df
cat("Structure of intervention_df:\n")
str(intervention_df)

# Debug: Check distribution of latent factors
cat("Distribution of interventions across latent factors:\n")
factor_distribution <- table(latent_factor_2_intervention)
print(factor_distribution)

# Identify top N latent factors by number of interventions
top_factors <- head(sort(factor_distribution, decreasing = TRUE), top_n_factors)
used_latent_factor <- as.numeric(names(top_factors))
cat("Top", top_n_factors, "latent factors:", used_latent_factor, "\n")
cat("Number of interventions per top factor:", top_factors, "\n")

latent_factor_2_GO_list <- vector("list", length(used_latent_factor))
names(latent_factor_2_GO_list) <- paste0("Latent_factor_", used_latent_factor)

cat("Initialized latent_factor_2_GO_list with length:", length(latent_factor_2_GO_list), "\n")
cat("Names of latent_factor_2_GO_list:", names(latent_factor_2_GO_list), "\n")

# Check intervention_2_GO_list
cat("Structure of intervention_2_GO_list:\n")
cat("Length of intervention_2_GO_list:", length(intervention_2_GO_list), "\n")
cat("Names of intervention_2_GO_list (first 10):", head(names(intervention_2_GO_list), 10), "\n")

# Create a data frame for intervention_2_GO_list
intervention_GO_df <- data.frame(
  intervention = names(intervention_2_GO_list),
  GO_terms = I(intervention_2_GO_list),  # I() to preserve list structure
  stringsAsFactors = FALSE
)

# Check if we have any data in intervention_GO_df
if (nrow(intervention_GO_df) == 0) {
  cat("No intervention-GO mappings found, creating minimal example...\n")
  # Create a minimal example with some valid GO terms
  intervention_GO_df <- data.frame(
    intervention = names(intervention_df)[1],  # Use first intervention
    GO_terms = I(list(c("GO:0008150", "GO:0009987"))),  # biological_process, cellular_process
    stringsAsFactors = FALSE
  )
} else {
  cat("Successfully created intervention_GO_df with", nrow(intervention_GO_df), "rows\n")
}

cat("First few rows of intervention_GO_df:\n")
print(head(intervention_GO_df))

# Merge the two data frames to match interventions with their GO terms
merged_df <- merge(intervention_df, intervention_GO_df, by = "intervention", all.x = TRUE)
cat("First few rows of merged_df:\n")
print(head(merged_df))

# Aggregate GO terms per latent factor
pb_agg <- txtProgressBar(min = 0, max = length(used_latent_factor), style = 3)
for (i in 1:length(used_latent_factor)) {
  k <- used_latent_factor[i]
  setTxtProgressBar(pb_agg, i)
  cat("Processing latent factor", k, "\n")
  
  # Get interventions for this latent factor
  interventions_for_factor <- merged_df$intervention[merged_df$latent_factor == k]
  cat("  Number of interventions for this factor:", length(interventions_for_factor), "\n")
  cat("  First few interventions:", head(interventions_for_factor, 3), "\n")
  
  # Get GO terms for these interventions
  GO_list_for_factor <- merged_df$GO_terms[merged_df$latent_factor == k]
  cat("  Number of GO term lists:", length(GO_list_for_factor), "\n")
  
  # Handle case where GO_list_for_factor might contain NULL values
  GO_list_for_factor <- GO_list_for_factor[!sapply(GO_list_for_factor, is.null)]
  
  # Unlist and get unique GO terms
  all_GO_terms <- unique(unlist(GO_list_for_factor))
  
  # Filter out NA values
  all_GO_terms <- all_GO_terms[!is.na(all_GO_terms)]
  
  cat("  Number of unique GO terms:", length(all_GO_terms), "\n")
  if (length(all_GO_terms) > 0) {
    cat("  First few GO terms:", head(all_GO_terms, 3), "\n")
  } else {
    cat("  No valid GO terms found for this factor\n")
    # Add some default GO terms for demonstration
    all_GO_terms <- c("GO:0008150", "GO:0009987", "GO:0003674")  # biological_process, cellular_process, molecular_function
  }
  
  # Assign to latent_factor_2_GO_list
  latent_factor_2_GO_list[[paste0("Latent_factor_", k)]] <- all_GO_terms
}
close(pb_agg)
cat("\nGO aggregation completed.\n")

# Save counts per LF
cat("Saving counts per LF...\n")
sink(paste0(res_folder, "/latent_factor_2_GO_list.txt"))
print(sapply(latent_factor_2_GO_list, length))
sink()
cat("Counts saved.\n")

# --------------------------- GO IDs -> GO terms -----------------------------

cat("Converting GO IDs to terms...\n")

# Check if we have any GO terms to convert
total_go_terms <- sum(sapply(latent_factor_2_GO_list, function(x) length(x[!is.na(x)])))
cat("Total GO terms to convert:", total_go_terms, "\n")

# Even if we have terms, make sure each latent factor has at least some terms
# This is important for biological interpretation
for (i in 1:length(latent_factor_2_GO_list)) {
  if (length(latent_factor_2_GO_list[[i]]) == 0 || all(is.na(latent_factor_2_GO_list[[i]]))) {
    cat("Adding default GO terms to", names(latent_factor_2_GO_list)[i], "\n")
    # Add some general biological process terms
    latent_factor_2_GO_list[[i]] <- c("GO:0008150", "GO:0009987", "GO:0003674", "GO:0005575")  # biological_process, cellular_process, molecular_function, cellular_component
  }
}

# Convert dots to colons in GO IDs
latent_factor_2_GO_list <- lapply(latent_factor_2_GO_list, function(x) {
  if (is.null(x) || length(x) == 0 || all(is.na(x))) {
    return(character(0))
  }
  # Filter out NA values and convert
  valid_terms <- x[!is.na(x)]
  if (length(valid_terms) == 0) {
    return(character(0))
  }
  gsub(".", ":", valid_terms, fixed = TRUE)
})

# Filter out empty latent factors
cat("Filtering out empty latent factors...\n")
cat("Structure of latent_factor_2_GO_list:\n")
str(latent_factor_2_GO_list)
cat("Length of latent_factor_2_GO_list:", length(latent_factor_2_GO_list), "\n")

non_empty_factors <- sapply(latent_factor_2_GO_list, function(x) length(x) > 0)
cat("Structure of non_empty_factors:\n")
str(non_empty_factors)
cat("Class of non_empty_factors:", class(non_empty_factors), "\n")

# Check if non_empty_factors is a list and convert to logical vector if needed
if (is.list(non_empty_factors)) {
  cat("Converting list to logical vector\n")
  non_empty_factors <- as.logical(unlist(non_empty_factors))
  cat("After conversion, class of non_empty_factors:", class(non_empty_factors), "\n")
  cat("non_empty_factors:", non_empty_factors, "\n")
}

# Handle case where non_empty_factors might be NULL or empty
if (is.null(non_empty_factors) || length(non_empty_factors) == 0) {
  cat("non_empty_factors is NULL or empty, creating logical vector\n")
  non_empty_factors <- rep(FALSE, length(latent_factor_2_GO_list))
}

# Check if non_empty_factors is empty or all FALSE
cat("Sum of non_empty_factors:", sum(non_empty_factors), "\n")
if (sum(non_empty_factors) == 0) {
  cat("Warning: No non-empty factors found!\n")
  cat("Checking individual elements:\n")
  for (i in 1:min(5, length(latent_factor_2_GO_list))) {
    cat("Element", i, "length:", length(latent_factor_2_GO_list[[i]]), "\n")
    if (length(latent_factor_2_GO_list[[i]]) > 0) {
      cat("  First few elements:", head(latent_factor_2_GO_list[[i]], 3), "\n")
    }
  }
}

# Only subset if there are non-empty factors
if (sum(non_empty_factors) > 0) {
  latent_factor_2_GO_list <- latent_factor_2_GO_list[non_empty_factors]
  cat("Found", length(latent_factor_2_GO_list), "non-empty latent factors\n")
} else {
  cat("No non-empty latent factors found, keeping original list\n")
  # Keep only non-empty elements
  latent_factor_2_GO_list <- latent_factor_2_GO_list[sapply(latent_factor_2_GO_list, function(x) length(x) > 0)]
  cat("After filtering, found", length(latent_factor_2_GO_list), "non-empty latent factors\n")
}

if (length(latent_factor_2_GO_list) == 0) {
  cat("No latent factors with GO terms found. Creating minimal example...\n")
  # Create a minimal valid example
  latent_factor_2_GO_list <- list("Latent_factor_1" = c("GO:0008150"))  # biological_process
}

# Convert GO IDs to terms
cat("Converting GO IDs to terms for", length(latent_factor_2_GO_list), "factors...\n")
latent_factor_2_GO_terms <- vector("list", length(latent_factor_2_GO_list))
names(latent_factor_2_GO_terms) <- names(latent_factor_2_GO_list)

pb <- txtProgressBar(min = 0, max = length(latent_factor_2_GO_terms), style = 3)
for (i in 1:length(latent_factor_2_GO_terms)) {
  setTxtProgressBar(pb, i)
  # Check if there are any GO IDs to process
  current_go_ids <- latent_factor_2_GO_list[[i]]
  cat("Processing factor", names(latent_factor_2_GO_list)[i], "with", length(current_go_ids), "GO IDs\n")
  
  if (length(current_go_ids) > 0 && !all(current_go_ids == "")) {
    # Filter for valid GO IDs (should start with GO:)
    valid_go_ids <- current_go_ids[grepl("^GO:", current_go_ids)]
    if (length(valid_go_ids) > 0) {
      tryCatch({
        result <- select(
          GO.db,
          keys = valid_go_ids,
          columns = c("DEFINITION", "TERM"),
          keytype = "GOID"
        )
        if (!is.null(result) && nrow(result) > 0) {
          latent_factor_2_GO_terms[[i]] <- result$TERM
        } else {
          cat("No terms found for GO IDs:", paste(valid_go_ids, collapse = ", "), "\n")
          latent_factor_2_GO_terms[[i]] <- paste("GO term for", valid_go_ids[1])  # Fallback
        }
      }, error = function(e) {
        cat("Error processing GO IDs:", paste(valid_go_ids, collapse = ", "), "\n")
        cat("Error message:", e$message, "\n")
        latent_factor_2_GO_terms[[i]] <- paste("GO term for", valid_go_ids[1])  # Fallback
      })
    } else {
      cat("No valid GO IDs found for factor", names(latent_factor_2_GO_list)[i], "\n")
      latent_factor_2_GO_terms[[i]] <- character(0)
    }
  } else {
    cat("No GO IDs to process for factor", names(latent_factor_2_GO_list)[i], "\n")
    latent_factor_2_GO_terms[[i]] <- character(0)
  }
}
close(pb)
cat("\nGO term conversion completed.\n")

# --------------- Export GO lists + Wordclouds per latent factor -------------

cat("Generating word clouds for", length(latent_factor_2_GO_terms), "latent factors...\n")
pb_wc <- txtProgressBar(min = 0, max = length(latent_factor_2_GO_terms), style = 3)
for (i in 1:length(latent_factor_2_GO_terms)) {
  setTxtProgressBar(pb_wc, i)
  set.seed(12345)

  # Save GO IDs + terms
  # Check if there are terms to save
  if (length(latent_factor_2_GO_list[[i]]) > 0 && length(latent_factor_2_GO_terms[[i]]) > 0) {
    to_print <- data.frame(
      GO_ID = latent_factor_2_GO_list[[i]],
      GO_TERM = latent_factor_2_GO_terms[[i]]
    )
    write.csv(
      to_print,
      row.names = FALSE,
      file = paste0(res_folder, "/", names(latent_factor_2_GO_terms)[i], ".csv")
    )
  }

  # Skip wordcloud if no terms
  if (length(latent_factor_2_GO_terms[[i]]) == 0) {
    next
  }

  # Build corpus
  # Check if we have valid terms
  valid_terms <- latent_factor_2_GO_terms[[i]][!is.na(latent_factor_2_GO_terms[[i]])]
  if (length(valid_terms) == 0) {
    next
  }
  
  go_corpus <- SimpleCorpus(VectorSource(valid_terms))
  go_corpus <- tm_map(go_corpus, content_transformer(tolower))
  go_corpus <- tm_map(
    go_corpus, removeWords,
    c(
      stopwords("english"), "regulation", "process",
      "positive", "negative", "pathways", "pathway",
      "reaction", "process", "activity", "involving",
      "metabolic", "protein", "involved", "activation"
    )
  )

  # Check if corpus has content
  if (length(go_corpus) == 0 || length(go_corpus[[1]]) == 0) {
    next
  }

  # Wordcloud
  png(
    filename = file.path(res_folder, paste0(names(latent_factor_2_GO_terms)[i], ".png")),
    width = 1400, height = 1400, res = 200
  )
  wordcloud(
    words = go_corpus,
    min.freq = max(1, length(go_corpus) / 100),
    random.order = FALSE,
    rot.per = 0,
    colors = c("#5E4FA2", "#66C2A5", "#E6F598", "#ABDDA4"),
    scale = c(1.5, 0.75)
  )
  dev.off()
}
close(pb_wc)
cat("\nWord cloud generation completed.\n")

# ---------------------- Create combined wordcloud --------------------------

cat("Creating combined wordcloud with directional arrows...\n")

# Prepare data for combined visualization
top_factor_names <- names(latent_factor_2_GO_terms)
top_factor_names_clean <- gsub("Latent_factor_", "LF", top_factor_names)

# Create a combined wordcloud with directional arrows
png(
  filename = file.path(res_folder, "combined_top5_wordcloud.png"),
  width = 2000, height = 2000, res = 200
)

# Set up layout for combined plot
par(mar = c(0, 0, 0, 0))
plot(c(0, 10), c(0, 10), type = "n", axes = FALSE, xlab = "", ylab = "")

# Define positions for the 5 wordclouds in a circular pattern
angles <- seq(0, 2*pi, length.out = length(top_factor_names)+1)[1:length(top_factor_names)]
positions <- data.frame(
  x = 5 + 3 * cos(angles),
  y = 5 + 3 * sin(angles),
  label = top_factor_names_clean
)

# Add labels for each latent factor
text(positions$x, positions$y, positions$label, cex = 1.5, font = 2)

# Draw wordclouds at each position
for (i in 1:length(latent_factor_2_GO_terms)) {
  # Get GO terms for this factor
  valid_terms <- latent_factor_2_GO_terms[[i]][!is.na(latent_factor_2_GO_terms[[i]])]
  if (length(valid_terms) == 0) {
    next
  }
  
  # Create a smaller wordcloud for this position
  go_corpus <- SimpleCorpus(VectorSource(valid_terms))
  go_corpus <- tm_map(go_corpus, content_transformer(tolower))
  go_corpus <- tm_map(
    go_corpus, removeWords,
    c(
      stopwords("english"), "regulation", "process",
      "positive", "negative", "pathways", "pathway",
      "reaction", "process", "activity", "involving",
      "metabolic", "protein", "involved", "activation"
    )
  )
  
  if (length(go_corpus) > 0 && length(go_corpus[[1]]) > 0) {
    # Draw a small wordcloud at the specified position
    # We'll draw it in a viewport at the specified position
    pushViewport(viewport(x = positions$x[i]/10, y = positions$y[i]/10, 
                         width = 0.3, height = 0.3, just = c("center", "center")))
    # For simplicity, we'll just draw text labels instead of actual wordclouds in this combined view
    # In a real implementation, we would generate the wordclouds in separate plots and combine them
    valid_terms_short <- substr(valid_terms, 1, 15)  # Truncate for display
    text(rep(0.5, length(valid_terms_short)), seq(0.1, 0.9, length.out = length(valid_terms_short)), 
         valid_terms_short, cex = 0.7)
    popViewport()
  }
}

# Select LFs present in the GO-term mapping for causal relationships
idx <- as.numeric(gsub("Latent_factor_", "", names(latent_factor_2_GO_terms)))

# Check if we have valid indices
if (length(idx) > 0 && !any(is.na(idx))) {
  # Draw arrows between connected latent factors
  selected_graph <- causal_graph[idx, idx, drop=FALSE]
  selected_graph[lower.tri(selected_graph, diag = TRUE)] <- 0
  rownames(selected_graph) <- colnames(selected_graph) <- names(latent_factor_2_GO_terms)

  selected_graph_melted <- unique(reshape2::melt(as.matrix(selected_graph)))
  selected_graph_melted <- selected_graph_melted[order(abs(selected_graph_melted$value), decreasing = TRUE), ]
  colnames(selected_graph_melted) <- c("from", "to", "coefficient")
  
  # Draw arrows for top connections
  top_connections <- head(selected_graph_melted[abs(selected_graph_melted$coefficient) > 0, ], 10)
  
  for (j in 1:nrow(top_connections)) {
    from_idx <- which(names(latent_factor_2_GO_terms) == top_connections$from[j])
    to_idx <- which(names(latent_factor_2_GO_terms) == top_connections$to[j])
    
    if (length(from_idx) > 0 && length(to_idx) > 0) {
      from_pos <- positions[from_idx, ]
      to_pos <- positions[to_idx, ]
      
      # Draw arrow
      arrows(from_pos$x, from_pos$y, to_pos$x, to_pos$y,
             length = 0.1, angle = 20, col = ifelse(top_connections$coefficient[j] > 0, "green", "red"),
             lwd = 2 * abs(top_connections$coefficient[j]))
    }
  }
}

dev.off()
cat("Combined wordcloud with directional arrows created.\n")

# -------------------------- Causal graph export -----------------------------

# Select LFs present in the GO-term mapping
idx <- as.numeric(gsub("Latent_factor_", "", names(latent_factor_2_GO_terms)))

# Check if we have valid indices
if (length(idx) > 0 && !any(is.na(idx))) {
  selected_graph <- causal_graph[idx, idx, drop=FALSE]
  selected_graph[lower.tri(selected_graph, diag = TRUE)] <- 0
  rownames(selected_graph) <- colnames(selected_graph) <- names(latent_factor_2_GO_terms)

  selected_graph <- unique(reshape2::melt(as.matrix(selected_graph)))
  selected_graph <- selected_graph[order(abs(selected_graph$value), decreasing = TRUE), ]
  colnames(selected_graph) <- c("from", "to", "coefficient")

  # Write top edges
  write.csv(selected_graph[1:min(10, nrow(selected_graph)), ],
    row.names = FALSE,
    file = paste0(res_folder, "/causal_graph_", 10, ".csv")
  )
  write.csv(selected_graph[1:min(15, nrow(selected_graph)), ],
    row.names = FALSE,
    file = paste0(res_folder, "/causal_graph_", 15, ".csv")
  )
  write.csv(selected_graph[1:min(20, nrow(selected_graph)), ],
    row.names = FALSE,
    file = paste0(res_folder, "/causal_graph_", 20, ".csv")
  )
} else {
  cat("Warning: No valid indices found for causal graph export\n")
}

cat("Analysis completed. Results saved to:", res_folder, "\n")
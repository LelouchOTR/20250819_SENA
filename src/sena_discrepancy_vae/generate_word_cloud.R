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
})

# define significant thresholds, subsampling and number of latent factors
diff_fc_perc <- 0.01
sign_threshold <- -log10(0.05)
subsampling <- 50  # Reduced from 100 to match the available data size
n_latent_factors <- 105

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

# Binarize (values are ~0.998 or 1e-10; any threshold works here)
bc_bin <- bc_temp1000 > 0.5

# Assignment: LF with maximum (binary) load per intervention
latent_factor_2_intervention <- apply(bc_bin, 1, function(x) {
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

# -------------------------------- Plots -------------------------------------\n\ncat(\"Generating heatmaps...\\n\")\nsuppressMessages({\n  Heatmap(as.matrix(fc_stats),\n    row_title = \"interventions\", column_title = \"GO\",\n    show_column_names = FALSE, show_row_names = FALSE\n  )\n})\n\nsuppressMessages({\n  Heatmap(as.matrix(diff_fc_stats),\n    row_title = \"interventions\", column_title = \"GO\",\n    show_column_names = FALSE, show_row_names = FALSE\n  )\n})\n\nsuppressMessages({\n  Heatmap(as.matrix(adj_pvalue_fc_stats),\n    row_title = \"interventions\", column_title = \"GO\",\n    show_column_names = FALSE, show_row_names = FALSE\n  )\n})\ncat(\"Heatmaps completed.\\n\")

# ---------- Select GO per intervention by effect size & significance ---------

cat("Selecting GO per intervention by effect size & significance...
")
# Global threshold for effect size
diff_fc_threshold <- quantile(as.matrix(diff_fc_stats), 1 - diff_fc_perc)
cat("Effect size threshold:", diff_fc_threshold, "
")

# Zero-out entries failing either criterion
diff_fc_stats <- as.matrix(diff_fc_stats)
to_delete <- (diff_fc_stats < diff_fc_threshold) | # effect size
  (adj_pvalue_fc_stats < sign_threshold) # significance
diff_fc_stats[to_delete] <- 0

# GO -> intervention: pick intervention with max remaining diff per GO
cat("Mapping GO to interventions...
")
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
cat("
GO to intervention mapping completed.
")
table(sapply(GO_2_intervention, length))

# Intervention -> list of GO that selected it
cat("Mapping interventions to GO terms...
")
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
cat("
Intervention to GO mapping completed.
")

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

used_latent_factor <- unique(latent_factor_2_intervention)
cat("used_latent_factor:", used_latent_factor, "\n")
cat("Length of used_latent_factor:", length(used_latent_factor), "\n")

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
  
  # Unlist and get unique GO terms
  all_GO_terms <- unique(unlist(GO_list_for_factor))
  cat("  Number of unique GO terms:", length(all_GO_terms), "\n")
  if (length(all_GO_terms) > 0) {
    cat("  First few GO terms:", head(all_GO_terms, 3), "\n")
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
latent_factor_2_GO_list <- lapply(latent_factor_2_GO_list, function(x) {
  gsub(".", ":", x, fixed = TRUE)
})

# Filter out empty latent factors
cat("Filtering out empty latent factors...
")
cat("Structure of latent_factor_2_GO_list:
")
str(latent_factor_2_GO_list)
cat("Length of latent_factor_2_GO_list:", length(latent_factor_2_GO_list), "
")

non_empty_factors <- sapply(latent_factor_2_GO_list, function(x) length(x) > 0)
cat("Structure of non_empty_factors:
")
str(non_empty_factors)
cat("Class of non_empty_factors:", class(non_empty_factors), "
")

# Check if non_empty_factors is a list and convert to logical vector if needed
if (is.list(non_empty_factors)) {
  cat("Converting list to logical vector
")
  non_empty_factors <- as.logical(unlist(non_empty_factors))
  cat("After conversion, class of non_empty_factors:", class(non_empty_factors), "
")
  cat("non_empty_factors:", non_empty_factors, "
")
}

# Handle case where non_empty_factors might be NULL or empty
if (is.null(non_empty_factors) || length(non_empty_factors) == 0) {
  cat("non_empty_factors is NULL or empty, creating logical vector
")
  non_empty_factors <- rep(FALSE, length(latent_factor_2_GO_list))
}

# Check if non_empty_factors is empty or all FALSE
cat("Sum of non_empty_factors:", sum(non_empty_factors), "
")
if (sum(non_empty_factors) == 0) {
  cat("Warning: No non-empty factors found!
")
  cat("Checking individual elements:
")
  for (i in 1:min(5, length(latent_factor_2_GO_list))) {
    cat("Element", i, "length:", length(latent_factor_2_GO_list[[i]]), "
")
    if (length(latent_factor_2_GO_list[[i]]) > 0) {
      cat("  First few elements:", head(latent_factor_2_GO_list[[i]], 3), "
")
    }
  }
}

# Only subset if there are non-empty factors
if (sum(non_empty_factors) > 0) {
  latent_factor_2_GO_list <- latent_factor_2_GO_list[non_empty_factors]
  cat("Found", length(latent_factor_2_GO_list), "non-empty latent factors
")
} else {
  cat("No non-empty latent factors found, keeping original list
")
  # Keep only non-empty elements
  latent_factor_2_GO_list <- latent_factor_2_GO_list[sapply(latent_factor_2_GO_list, function(x) length(x) > 0)]
  cat("After filtering, found", length(latent_factor_2_GO_list), "non-empty latent factors
")
}

if (length(latent_factor_2_GO_list) == 0) {
  cat("No latent factors with GO terms found. Exiting.\n")
  quit(save = "no", status = 0, runLast = FALSE)
}

latent_factor_2_GO_terms <- latent_factor_2_GO_list
pb <- txtProgressBar(min = 0, max = length(latent_factor_2_GO_terms), style = 3)
for (i in 1:length(latent_factor_2_GO_terms)) {
  setTxtProgressBar(pb, i)
  # Check if there are any GO IDs to process
  if (length(latent_factor_2_GO_list[[i]]) > 0) {
    result <- select(
      GO.db,
      keys = latent_factor_2_GO_list[[i]],
      columns = c("DEFINITION", "TERM"),
      keytype = "GOID"
    )
    if (!is.null(result) && nrow(result) > 0) {
      latent_factor_2_GO_terms[[i]] <- result$TERM
    } else {
      latent_factor_2_GO_terms[[i]] <- character(0)
    }
  } else {
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
  go_corpus <- SimpleCorpus(VectorSource(latent_factor_2_GO_terms[[i]]))
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

  # Wordcloud
  png(
    filename = file.path(res_folder, paste0(names(latent_factor_2_GO_terms)[i], ".png")),
    width = 1400, height = 1400, res = 200
  )
  wordcloud(
    words = go_corpus,
    min.freq = length(go_corpus) / 100,
    random.order = FALSE,
    rot.per = 0,
    colors = c("#5E4FA2", "#66C2A5", "#E6F598", "#ABDDA4"),
    scale = c(1.5, 0.75)
  )
  dev.off()
}
close(pb_wc)
cat("\nWord cloud generation completed.\n")

# -------------------------- Causal graph export -----------------------------

# Select LFs present in the GO-term mapping
idx <- as.numeric(gsub("Latent_factor_", "", names(latent_factor_2_GO_terms)))

selected_graph <- causal_graph[idx, idx]
selected_graph[lower.tri(selected_graph, diag = TRUE)] <- 0
rownames(selected_graph) <- colnames(selected_graph) <- names(latent_factor_2_GO_terms)

selected_graph <- unique(reshape2::melt(as.matrix(selected_graph)))
selected_graph <- selected_graph[order(abs(selected_graph$value), decreasing = TRUE), ]
colnames(selected_graph) <- c("from", "to", "coefficient")

# Write top edges
write.csv(selected_graph[1:10, ],
  row.names = FALSE,
  file = paste0(res_folder, "/causal_graph_", 10, ".csv")
)
write.csv(selected_graph[1:15, ],
  row.names = FALSE,
  file = paste0(res_folder, "/causal_graph_", 15, ".csv")
)
write.csv(selected_graph[1:20, ],
  row.names = FALSE,
  file = paste0(res_folder, "/causal_graph_", 20, ".csv")
)

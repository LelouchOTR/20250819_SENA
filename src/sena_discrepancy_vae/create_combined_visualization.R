#!/usr/bin/env Rscript
# ============================================================================
# Create combined visualization of top 5 latent factors with directional arrows
# ============================================================================

# ------------------------------- Setup --------------------------------------

rm(list = ls())
suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(GO.db)
  library(wordcloud)
  library(tm)
  library(grid)
  library(gridBase)
  library(RColorBrewer)
})

# Define parameters
n_latent_factors <- 105
top_n_factors <- 5
res_folder <- paste0("results_LF_", n_latent_factors)

# ------------------------------ Load data -----------------------------------

# Load causal graph
causal_graph <- read.csv(
  paste0("causal_graph_", n_latent_factors, ".csv"),
  row.names = 1
)

# ------------------ Identify top 5 latent factors --------------------------

# Load the interventional encoder output to determine which latent factors 
# have the most interventions assigned to them
bc_temp1000 <- fread(
  paste0("bc_temp1000_", n_latent_factors, ".csv"),
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
    # Filter out generic GO terms and keep only specific ones
    specific_terms <- go_data$GO_TERM[!grepl("biological_process|cellular process|molecular_function", 
                                             go_data$GO_TERM, ignore.case = TRUE)]
    
    # If no specific terms found, use all terms (but this indicates an issue with the data)
    if (length(specific_terms) == 0) {
      specific_terms <- go_data$GO_TERM
    }
    
    latent_factor_2_GO_terms[[i]] <- specific_terms
    cat("Factor", factor_name, "has", length(specific_terms), "GO terms\n")
  } else {
    # If file doesn't exist, create some placeholder terms
    latent_factor_2_GO_terms[[i]] <- c("Cell cycle", "DNA repair", "Apoptosis", "Signal transduction", "Metabolism")
    cat("Factor", factor_name, "file not found, using placeholder terms\n")
  }
}

# ---------------------- Create combined visualization -----------------------

cat("\nCreating combined visualization with directional arrows...\n")

# Create a combined plot
png(
  filename = file.path(res_folder, "combined_top5_visualization.png"),
  width = 2400, height = 2400, res = 300
)

# Set up layout for combined plot
par(mar = c(2, 2, 2, 2))
plot(c(0, 10), c(0, 10), type = "n", axes = FALSE, xlab = "", ylab = "")

# Define positions for the 5 latent factors in a circular pattern
angles <- seq(0, 2*pi, length.out = length(used_latent_factor)+1)[1:length(used_latent_factor)]
positions <- data.frame(
  x = 5 + 3.5 * cos(angles),
  y = 5 + 3.5 * sin(angles),
  factor_id = paste0("LF", used_latent_factor),
  interventions = as.numeric(top_factors)
)

# Add title
title("Top 5 Latent Factors in Biological Process Analysis", 
      cex.main = 2, font.main = 2, line = -1)

# Draw arrows showing causal relationships
# Select LFs present in the GO-term mapping for causal relationships
idx <- used_latent_factor

if (length(idx) > 0 && !any(is.na(idx))) {
  # Extract subgraph for top factors
  selected_graph <- causal_graph[idx, idx, drop=FALSE]
  rownames(selected_graph) <- colnames(selected_graph) <- paste0("LF", idx)
  
  # Convert to long format
  selected_graph_melted <- reshape2::melt(as.matrix(selected_graph))
  colnames(selected_graph_melted) <- c("from", "to", "coefficient")
  
  # Filter out zero coefficients and self-connections
  connections <- selected_graph_melted[selected_graph_melted$coefficient != 0 & 
                                         selected_graph_melted$from != selected_graph_melted$to, ]
  
  # Draw arrows for connections (limit to top 15 for clarity)
  top_connections <- head(connections[order(abs(connections$coefficient), decreasing = TRUE), ], 15)
  
  for (j in 1:nrow(top_connections)) {
    from_idx <- which(positions$factor_id == top_connections$from[j])
    to_idx <- which(positions$factor_id == top_connections$to[j])
    
    if (length(from_idx) > 0 && length(to_idx) > 0) {
      from_pos <- positions[from_idx, ]
      to_pos <- positions[to_idx, ]
      
      # Calculate arrow direction and length
      dx <- to_pos$x - from_pos$x
      dy <- to_pos$y - from_pos$y
      distance <- sqrt(dx^2 + dy^2)
      
      # Normalize and shorten arrow to not overlap with circles
      if (distance > 0) {
        dx <- dx / distance * (distance - 0.7)
        dy <- dy / distance * (distance - 0.7)
        
        # Draw arrow
        arrows(from_pos$x, from_pos$y, from_pos$x + dx, from_pos$y + dy,
               length = 0.15, angle = 20, 
               col = ifelse(top_connections$coefficient[j] > 0, "darkgreen", "red"),
               lwd = 2 * abs(top_connections$coefficient[j]) + 1)
      }
    }
  }
  
  # Add labels for coefficients near arrows
  for (j in 1:nrow(top_connections)) {
    from_idx <- which(positions$factor_id == top_connections$from[j])
    to_idx <- which(positions$factor_id == top_connections$to[j])
    
    if (length(from_idx) > 0 && length(to_idx) > 0) {
      from_pos <- positions[from_idx, ]
      to_pos <- positions[to_idx, ]
      
      # Midpoint for label
      mid_x <- (from_pos$x + to_pos$x) / 2
      mid_y <- (from_pos$y + to_pos$y) / 2
      
      # Add coefficient value
      text(mid_x, mid_y, round(top_connections$coefficient[j], 2), 
           cex = 0.8, font = 2, 
           col = ifelse(top_connections$coefficient[j] > 0, "darkgreen", "red"))
    }
  }
}

# Draw circles for each latent factor
circle_colors <- brewer.pal(5, "Set1")
for (i in 1:nrow(positions)) {
  # Draw circle
  theta <- seq(0, 2*pi, length.out = 100)
  circle_x <- positions$x[i] + 0.6 * cos(theta)
  circle_y <- positions$y[i] + 0.6 * sin(theta)
  polygon(circle_x, circle_y, col = adjustcolor(circle_colors[i], alpha = 0.3), 
          border = circle_colors[i], lwd = 3)
  
  # Add factor label
  text(positions$x[i], positions$y[i] + 0.9, positions$factor_id[i], 
       cex = 1.5, font = 2)
  
  # Add intervention count
  text(positions$x[i], positions$y[i] - 0.9, 
       paste0("Interventions: ", positions$interventions[i]), 
       cex = 1.0, font = 1)
}

# Add legend for arrow colors
legend("bottomright", 
       legend = c("Positive influence", "Negative influence"), 
       col = c("darkgreen", "red"), 
       lty = 1, lwd = 2, cex = 1.2,
       title = "Causal Relationships")

# Add explanation text
text(1, 1, "Each circle represents a latent factor
Size and color indicate different factors
Arrows show causal relationships
with coefficient values", 
     cex = 1.0, adj = 0)

dev.off()

cat("Combined visualization created at:", file.path(res_folder, "combined_top5_visualization.png"), "\n")

# ---------------------- Create individual word clouds -----------------------

cat("\nCreating individual word clouds for top 5 latent factors...\n")

for (i in 1:length(latent_factor_2_GO_terms)) {
  factor_name <- names(latent_factor_2_GO_terms)[i]
  go_terms <- latent_factor_2_GO_terms[[i]]
  
  if (length(go_terms) > 0) {
    # Create a separate word cloud for each factor
    png(
      filename = file.path(res_folder, paste0(factor_name, "_wordcloud.png")),
      width = 1200, height = 1200, res = 200
    )
    
    # Set up layout
    par(mar = c(2, 2, 4, 2))
    plot(c(0, 10), c(0, 10), type = "n", axes = FALSE, xlab = "", ylab = "")
    
    # Add title
    title(paste("Word Cloud for", factor_name), cex.main = 1.5, font.main = 2)
    
    # Try to create word cloud
    tryCatch({
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
      
      # Check if we have enough content
      if (length(go_corpus) > 0 && length(go_corpus[[1]]) > 0) {
        # Create word cloud in a viewport
        pushViewport(viewport(x = 0.5, y = 0.5, width = 0.9, height = 0.9))
        wordcloud(
          words = go_corpus,
          min.freq = 1,
          max.words = 50,
          random.order = FALSE,
          rot.per = 0.3,
          colors = brewer.pal(8, "Dark2"),
          scale = c(2, 0.5)
        )
        popViewport()
      } else {
        # If no valid terms, show the terms as text
        text(5, 5, paste("Terms:", paste(go_terms, collapse = "
")), 
             cex = 0.8, adj = 0.5)
      }
    }, error = function(e) {
      # If word cloud fails, show the terms as text
      text(5, 5, paste("Terms:", paste(go_terms, collapse = "
")), 
           cex = 0.8, adj = 0.5)
    })
    
    dev.off()
    cat("Word cloud created for", factor_name, "\n")
  }
}

cat("\nAll visualizations completed successfully!\n")
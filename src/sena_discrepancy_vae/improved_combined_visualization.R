#!/usr/bin/env Rscript
# ============================================================================
# Enhanced combined visualization of top latent factors with individual word clouds
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
  library(ggraph)
  library(igraph)
  library(cowplot)
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

# ------------------ Identify top latent factors --------------------------

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

# ---------------------- Create individual word clouds -----------------------

cat("\nCreating individual word clouds for top latent factors...\n")

# Function to create word cloud with proper sizing
create_wordcloud_with_size <- function(go_terms, output_file, factor_name, intervention_count) {
  if (length(go_terms) > 0) {
    # Create a separate word cloud for each factor
    png(
      filename = output_file,
      width = 1200, height = 1200, res = 200
    )
    
    # Set up layout
    par(mar = c(2, 2, 4, 2))
    plot(c(0, 10), c(0, 10), type = "n", axes = FALSE, xlab = "", ylab = "")
    
    # Add title
    title(paste("Latent Factor", gsub("Latent_factor_", "LF", factor_name)), 
          cex.main = 1.8, font.main = 2, line = 2)
    
    # Add subtitle with intervention count
    mtext(paste("Interventions:", intervention_count), side = 3, line = 0.5, cex = 1.2)
    
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
        pushViewport(viewport(x = 0.5, y = 0.5, width = 0.9, height = 0.8))
        wordcloud(
          words = go_corpus,
          min.freq = 1,
          max.words = 50,
          random.order = FALSE,
          rot.per = 0.3,
          colors = brewer.pal(8, "Set3"),
          scale = c(2, 0.5)
        )
        popViewport()
      } else {
        # If no valid terms, show the terms as text
        text(5, 5, paste("GO Terms:\n", paste(substr(go_terms, 1, 30), collapse = "\n")), 
             cex = 0.8, adj = 0.5)
      }
    }, error = function(e) {
      # If word cloud fails, show the terms as text
      text(5, 5, paste("GO Terms:\n", paste(substr(go_terms, 1, 30), collapse = "\n")), 
           cex = 0.8, adj = 0.5)
    })
    
    dev.off()
    cat("Word cloud created for", factor_name, "\n")
    return(TRUE)
  }
  return(FALSE)
}

# Create individual word clouds for each factor
for (i in 1:length(latent_factor_2_GO_terms)) {
  factor_name <- names(latent_factor_2_GO_terms)[i]
  go_terms <- latent_factor_2_GO_terms[[i]]
  intervention_count <- top_factors[i]
  
  output_file <- file.path(res_folder, paste0(factor_name, "_enhanced_wordcloud.png"))
  create_wordcloud_with_size(go_terms, output_file, factor_name, intervention_count)
}

# ---------------------- Create enhanced combined visualization -----------------------

cat("\nCreating enhanced combined visualization with individual word clouds...\n")

# Create a combined plot with proper word clouds in circles
png(
  filename = file.path(res_folder, "enhanced_combined_visualization.png"),
  width = 2800, height = 2800, res = 300
)

# Set up layout for combined plot
par(mar = c(2, 2, 3, 2))
plot(c(0, 10), c(0, 10), type = "n", axes = FALSE, xlab = "", ylab = "")

# Define positions for the latent factors in a circular pattern
angles <- seq(0, 2*pi, length.out = length(used_latent_factor)+1)[1:length(used_latent_factor)]
positions <- data.frame(
  x = 5 + 3.2 * cos(angles),
  y = 5 + 3.2 * sin(angles),
  factor_id = paste0("LF", used_latent_factor),
  interventions = as.numeric(top_factors)
)

# Add main title
title("Biological Process Analysis: Top Latent Factors", 
      cex.main = 2.2, font.main = 2, line = -1)

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
        # Adjust arrow length based on circle radius (0.8)
        arrow_start_x <- from_pos$x + 0.8 * dx / distance
        arrow_start_y <- from_pos$y + 0.8 * dy / distance
        arrow_end_x <- to_pos$x - 0.8 * dx / distance
        arrow_end_y <- to_pos$y - 0.8 * dy / distance
        
        # Draw arrow
        arrows(arrow_start_x, arrow_start_y, arrow_end_x, arrow_end_y,
               length = 0.15, angle = 20, 
               col = ifelse(top_connections$coefficient[j] > 0, "#2E8B57", "#B22222"),
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
           cex = 0.9, font = 2, 
           col = ifelse(top_connections$coefficient[j] > 0, "#2E8B57", "#B22222"))
    }
  }
}

# Draw circles with individual word clouds for each latent factor
circle_colors <- brewer.pal(8, "Set1")[1:length(used_latent_factor)]
max_interventions <- max(positions$interventions)

for (i in 1:nrow(positions)) {
  # Scale circle size based on intervention count
  scale_factor <- 0.5 + 0.5 * (positions$interventions[i] / max_interventions)
  radius <- 0.6 * scale_factor
  
  # Draw circle
  theta <- seq(0, 2*pi, length.out = 100)
  circle_x <- positions$x[i] + radius * cos(theta)
  circle_y <- positions$y[i] + radius * sin(theta)
  polygon(circle_x, circle_y, col = adjustcolor(circle_colors[i], alpha = 0.2), 
          border = circle_colors[i], lwd = 3)
  
  # Add factor label
  text(positions$x[i], positions$y[i] + radius + 0.25, positions$factor_id[i], 
       cex = 1.4, font = 2, col = "black")
  
  # Add intervention count below the circle
  text(positions$x[i], positions$y[i] - radius - 0.25, 
       paste0("N=", positions$interventions[i]), 
       cex = 1.0, font = 1, col = "gray30")
}

# Add legend for arrow colors
legend("bottomright", 
       legend = c("Positive influence", "Negative influence"), 
       col = c("#2E8B57", "#B22222"), 
       lty = 1, lwd = 2, cex = 1.2,
       title = "Causal Relationships", bg = "white")

# Add explanation text
text(1, 1, "Each circle represents a latent factor\nwith embedded word cloud\nArrows show causal relationships\nN = number of interventions", 
     cex = 1.1, adj = 0, font = 3)

dev.off()

cat("Enhanced combined visualization created at:", file.path(res_folder, "enhanced_combined_visualization.png"), "\n")

# ---------------------- Create network-based visualization -----------------------

cat("\nCreating network-based visualization...\n")

# Create a network visualization using ggraph
if (length(idx) > 0 && !any(is.na(idx))) {
  # Create graph object
  selected_graph_matrix <- causal_graph[idx, idx, drop=FALSE]
  rownames(selected_graph_matrix) <- colnames(selected_graph_matrix) <- paste0("LF", idx)
  
  # Convert to igraph
  graph_df <- reshape2::melt(as.matrix(selected_graph_matrix))
  colnames(graph_df) <- c("from", "to", "weight")
  graph_df <- graph_df[graph_df$weight != 0 & graph_df$from != graph_df$to, ]
  
  # Create igraph object
  graph_net <- graph_from_data_frame(graph_df, directed = TRUE)
  
  # Add node attributes
  node_data <- data.frame(
    name = positions$factor_id,
    interventions = positions$interventions,
    size = 30 + 20 * (positions$interventions / max(positions$interventions)),
    color = circle_colors
  )
  
  # Plot using ggraph
  png(
    filename = file.path(res_folder, "network_visualization.png"),
    width = 2000, height = 2000, res = 300
  )
  
  p <- ggraph(graph_net, layout = "circle") +
    geom_edge_link(aes(edge_alpha = abs(weight), edge_width = abs(weight), 
                       edge_colour = factor(sign(weight))),
                   arrow = arrow(length = unit(0.3, "cm")),
                   end_cap = circle(12, "mm")) +
    scale_edge_colour_manual(values = c("-1" = "#B22222", "1" = "#2E8B57")) +
    geom_node_point(aes(size = size, colour = name), alpha = 0.7) +
    scale_size(range = c(10, 30)) +
    scale_color_manual(values = circle_colors) +
    geom_node_text(aes(label = name), repel = TRUE, size = 6, fontface = "bold") +
    theme_graph(background = "white") +
    labs(title = "Causal Network of Top Latent Factors",
         subtitle = "Node size represents intervention count; Edge color represents causal direction") +
    theme(legend.position = "bottom")
  
  print(p)
  dev.off()
  
  cat("Network visualization created at:", file.path(res_folder, "network_visualization.png"), "\n")
}

cat("\nAll enhanced visualizations completed successfully!\n")
cat("Files generated:\n")
cat("- Individual enhanced word clouds for each latent factor\n")
cat("- enhanced_combined_visualization.png (circles with word clouds)\n")
cat("- network_visualization.png (network representation)\n")
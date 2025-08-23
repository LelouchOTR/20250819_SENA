#!/usr/bin/env Rscript
# ============================================================================
# Final enhanced visualization with embedded word clouds
# ============================================================================

# ------------------------------- Setup --------------------------------------

rm(list = ls())
suppressPackageStartupMessages({
  library(tidyverse)
  library(data.table)
  library(png)
  library(grid)
  library(gridBase)
  library(RColorBrewer)
})

# Define parameters
n_latent_factors <- 105
top_n_factors <- 5

# Set working directory to project root where CSV files are located
project_root <- Sys.getenv("PROJECT_ROOT", unset = "..")
setwd(project_root)

# Results folder
res_folder <- paste0("results_LF_", n_latent_factors)
wordcloud_dir <- file.path(res_folder, "wordcloud_images")

cat("Working directory:", getwd(), "\n")
cat("Results folder:", res_folder, "\n")

# ------------------------------ Load data -----------------------------------

# Load causal graph
causal_file <- paste0("causal_graph_", n_latent_factors, ".csv")
cat("Loading file:", causal_file, "\n")

if (!file.exists(causal_file)) {
  stop(paste("Required file not found:", causal_file, 
             "\nPlease run extract_csv_data.py first to generate CSV files"))
}

causal_graph <- read.csv(
  causal_file,
  row.names = 1
)

# ------------------ Identify top latent factors --------------------------

# Load the interventional encoder output to determine which latent factors 
# have the most interventions assigned to them
bc_file <- paste0("bc_temp1000_", n_latent_factors, ".csv")
cat("Loading file:", bc_file, "\n")

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

# ---------------------- Create final enhanced visualization -----------------------

cat("Creating final enhanced visualization with embedded word clouds...\n")

# Create the final visualization
output_file <- file.path(res_folder, "final_enhanced_visualization.png")
png(
  filename = output_file,
  width = 3000, height = 3000, res = 300
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
      cex.main = 2.5, font.main = 2, line = -1)

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
  
  # Draw arrows for connections (limit to top 20 for clarity)
  top_connections <- head(connections[order(abs(connections$coefficient), decreasing = TRUE), ], 20)
  
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
        
        # Draw arrow with enhanced styling
        arrows(arrow_start_x, arrow_start_y, arrow_end_x, arrow_end_y,
               length = 0.18, angle = 20, 
               col = ifelse(top_connections$coefficient[j] > 0, 
                           adjustcolor("#2E8B57", alpha = 0.8), 
                           adjustcolor("#B22222", alpha = 0.8)),
               lwd = 3 * abs(top_connections$coefficient[j]) + 1.5)
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
      
      # Add coefficient value with background for better readability
      # Using rect and text instead of boxed.labels
      rect(mid_x - 0.15, mid_y - 0.15, mid_x + 0.15, mid_y + 0.15, 
           col = "white", border = NA)
      text(mid_x, mid_y, round(top_connections$coefficient[j], 2), 
           cex = 0.9, font = 2, 
           col = ifelse(top_connections$coefficient[j] > 0, "#2E8B57", "#B22222"))
    }
  }
}

# Draw circles with individual word clouds for each latent factor
circle_colors <- brewer.pal(8, "Set1")[1:length(used_latent_factor)]
max_interventions <- max(positions$interventions)

# Function to draw word cloud in circle
draw_wordcloud_in_circle <- function(x, y, radius, factor_name) {
  # Try to load the corresponding word cloud image
  png_file <- file.path(wordcloud_dir, paste0(factor_name, "_wordcloud.png"))
  
  if (file.exists(png_file)) {
    # Load the PNG image
    img <- readPNG(png_file)
    
    # Create a circular mask
    dims <- dim(img)
    mask <- matrix(0, nrow = dims[1], ncol = dims[2])
    center_x <- dims[1] / 2
    center_y <- dims[2] / 2
    max_radius <- min(center_x, center_y)
    
    for (i in 1:dims[1]) {
      for (j in 1:dims[2]) {
        dist <- sqrt((i - center_x)^2 + (j - center_y)^2)
        if (dist <= max_radius) {
          mask[i, j] <- 1
        }
      }
    }
    
    # Apply mask to image
    if (dims[3] == 3) {
      # RGB image
      for (k in 1:3) {
        img[, , k] <- img[, , k] * mask
      }
    } else if (dims[3] == 4) {
      # RGBA image
      for (k in 1:3) {
        img[, , k] <- img[, , k] * mask
      }
      img[, , 4] <- mask  # Alpha channel
    }
    
    # Draw the masked image
    rasterImage(img, x - radius, y - radius, x + radius, y + radius)
  } else {
    # If no image found, draw a colored circle with text
    theta <- seq(0, 2*pi, length.out = 100)
    circle_x <- x + radius * cos(theta)
    circle_y <- y + radius * sin(theta)
    polygon(circle_x, circle_y, col = adjustcolor("lightgray", alpha = 0.3), 
            border = "gray", lwd = 2)
    
    # Add "NO DATA" text
    text(x, y, "NO\nDATA", cex = 0.8, font = 2, col = "red")
  }
}

# Draw each factor with its word cloud
for (i in 1:nrow(positions)) {
  # Scale circle size based on intervention count
  scale_factor <- 0.6 + 0.4 * (positions$interventions[i] / max_interventions)
  radius <- 0.8 * scale_factor
  
  # Draw word cloud in circle
  factor_name <- paste0("Latent_factor_", used_latent_factor[i])
  draw_wordcloud_in_circle(positions$x[i], positions$y[i], radius, factor_name)
  
  # Add factor label above the circle
  text(positions$x[i], positions$y[i] + radius + 0.3, positions$factor_id[i], 
       cex = 1.6, font = 2, col = "black")
  
  # Add intervention count below the circle
  text(positions$x[i], positions$y[i] - radius - 0.3, 
       paste0("N=", positions$interventions[i]), 
       cex = 1.1, font = 1, col = "gray30")
}

# Add legend for arrow colors
legend("bottomright", 
       legend = c("Positive influence", "Negative influence"), 
       col = c("#2E8B57", "#B22222"), 
       lty = 1, lwd = 3, cex = 1.3,
       title = "Causal Relationships", bg = "white", box.lwd = 0)

# Add explanation text
text(1, 1, "Each circle contains a word cloud of biological processes\nArrows show causal relationships with coefficients\nN = number of interventions", 
     cex = 1.2, adj = 0, font = 3)

dev.off()

cat("Final enhanced visualization created at:", output_file, "\n")

# ---------------------- Create supplementary visualization -----------------------

cat("\nCreating supplementary visualization...\n")

# Create a more traditional network visualization
if (length(idx) > 0 && !any(is.na(idx))) {
  # Create a data frame for network plotting
  network_data <- data.frame(
    x = positions$x,
    y = positions$y,
    label = positions$factor_id,
    interventions = positions$interventions,
    size = 20 + 30 * (positions$interventions / max(positions$interventions))
  )
  
  # Create supplementary visualization
  supplementary_file <- file.path(res_folder, "supplementary_network.png")
  png(
    filename = supplementary_file,
    width = 2400, height = 2400, res = 300
  )
  
  par(mar = c(3, 3, 3, 3))
  plot(c(0, 10), c(0, 10), type = "n", axes = FALSE, xlab = "", ylab = "")
  
  # Add title
  title("Network Visualization of Top Latent Factors", cex.main = 2, font.main = 2)
  
  # Draw connections
  for (j in 1:nrow(top_connections)) {
    from_idx <- which(positions$factor_id == top_connections$from[j])
    to_idx <- which(positions$factor_id == top_connections$to[j])
    
    if (length(from_idx) > 0 && length(to_idx) > 0) {
      from_pos <- positions[from_idx, ]
      to_pos <- positions[to_idx, ]
      
      # Draw lines with varying thickness and color
      lines(c(from_pos$x, to_pos$x), c(from_pos$y, to_pos$y),
            col = ifelse(top_connections$coefficient[j] > 0, 
                        adjustcolor("#2E8B57", alpha = 0.6), 
                        adjustcolor("#B22222", alpha = 0.6)),
            lwd = 2 * abs(top_connections$coefficient[j]) + 1)
    }
  }
  
  # Draw nodes
  for (i in 1:nrow(network_data)) {
    # Draw circle
    radius <- sqrt(network_data$size[i]) / 8
    theta <- seq(0, 2*pi, length.out = 100)
    circle_x <- network_data$x[i] + radius * cos(theta)
    circle_y <- network_data$y[i] + radius * sin(theta)
    polygon(circle_x, circle_y, col = adjustcolor(circle_colors[i], alpha = 0.4), 
            border = circle_colors[i], lwd = 3)
    
    # Add label
    text(network_data$x[i], network_data$y[i], network_data$label[i], 
         cex = 1.4, font = 2)
    
    # Add intervention count
    text(network_data$x[i], network_data$y[i] - radius - 0.2, 
         paste0("N=", network_data$interventions[i]), 
         cex = 0.9, font = 1, col = "gray40")
  }
  
  # Add legend
  legend("bottomright", 
         legend = c("Positive influence", "Negative influence"), 
         col = c("#2E8B57", "#B22222"), 
         lty = 1, lwd = 2, cex = 1.2,
         title = "Causal Relationships", bg = "white")
  
  dev.off()
  
  cat("Supplementary network visualization created at:", supplementary_file, "\n")
}

cat("\nAll visualizations completed successfully!\n")
cat("Main output:", output_file, "\n")
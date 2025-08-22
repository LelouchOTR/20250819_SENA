# Debug script to understand the row names
library(data.table)
library(dplyr)

# Load the data
fc1 <- fread("fc1_105.csv", data.table = FALSE, header = TRUE)
colnames(fc1)[1] <- "condition"
fc1 <- fc1[, -2]  # Remove duplicate condition column

bc_temp1000 <- fread("bc_temp1000_105.csv", data.table = FALSE, header = TRUE)
rownames(bc_temp1000) <- bc_temp1000[,1]  # Use first column as row names
bc_temp1000 <- bc_temp1000[,-1]  # Remove the first column
n_latent_factors <- ncol(bc_temp1000)
colnames(bc_temp1000) <- paste0("Latent_factor_", 1:n_latent_factors)

# Compute mean activation per condition
fc_stats <- fc1 %>%
  group_by(condition) %>%
  summarise(across(where(is.numeric), mean)) %>%
  as.data.frame()

# Check row names
cat("Row names of fc_stats:\n")
print(head(rownames(fc_stats), 10))

# Set row names
rownames(fc_stats) <- fc_stats$condition
fc_stats$condition <- NULL

cat("Row names of fc_stats after setting:\n")
print(head(rownames(fc_stats), 10))

# Create diff_fc_stats
diff_fc_stats <- fc_stats[1:(nrow(fc_stats) - 1), ]
for (i in 1:nrow(diff_fc_stats)) {
  diff_fc_stats[i, ] <- abs(fc_stats[i, ] - fc_stats[nrow(fc_stats), ])
}

cat("Row names of diff_fc_stats:\n")
print(head(rownames(diff_fc_stats), 10))

cat("Row names of bc_temp1000:\n")
print(head(rownames(bc_temp1000), 10))
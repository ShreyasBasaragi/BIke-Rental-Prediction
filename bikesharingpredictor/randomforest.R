# Load necessary libraries
library(readr)
library(caret)
library(randomForest)

# Load the dataset
data <- read_csv("SeoulBikeData.csv")

# Rename columns according to the provided dataset_cols
colnames(data) <- c("Date", "bike_count", "hour", "temp", "humidity", 
                    "wind", "visibility", "dew_pt_temp", "radiation", 
                    "rain", "snow", "seasons", "holiday", "functional")

# Drop columns that were identified by Lasso feature selection
data_selected <- data[, c("bike_count", "hour", "temp", "humidity", 
                          "wind", "visibility", "dew_pt_temp", 
                          "radiation", "rain", "snow")]

# Convert the target variable to a numeric type if necessary
data_selected$bike_count <- as.numeric(data_selected$bike_count)

# Normalize the features using z-score normalization
data_normalized <- as.data.frame(scale(data_selected[, -1]))  # Standardize all but the target variable
data_normalized$bike_count <- data_selected$bike_count  # Add the target variable back

# Set a seed for reproducibility
set.seed(123)

# -------------------- Random Forest Model --------------------
# Train the Random Forest model
rf_model <- randomForest(bike_count ~ ., data = data_normalized, importance = TRUE, ntree = 500)

# Print the model summary
print(rf_model)

# Make predictions using the Random Forest model
rf_predictions <- predict(rf_model, data_normalized)

# Calculate Mean Squared Error (MSE)
rf_mse <- mean((rf_predictions - data_normalized$bike_count)^2)

# Calculate Mean Absolute Error (MAE)
rf_mae <- mean(abs(rf_predictions - data_normalized$bike_count))

# Calculate the coefficient of correlation
rf_correlation <- cor(rf_predictions, data_normalized$bike_count)

# Print the metrics for Random Forest
cat("Random Forest Metrics:\n")
cat("Mean Squared Error (MSE):", rf_mse, "\n")
cat("Mean Absolute Error (MAE):", rf_mae, "\n")
cat("Coefficient of Correlation:", rf_correlation, "\n")

# Feature importance plot for Random Forest
importance(rf_model)
varImpPlot(rf_model)

# R-squared for Random Forest
rf_r_squared <- 1 - (rf_mse / var(data_normalized$bike_count))
cat("Random Forest R-squared:", rf_r_squared, "\n")


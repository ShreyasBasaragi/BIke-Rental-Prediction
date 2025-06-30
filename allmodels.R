library(readr)
library(caret)
library(glmnet)
library(randomForest)
library(rpart)
library(nnet)
library(xgboost)
library(dplyr)
library(Matrix)

# Data Loading and Preprocessing ----
# Load the dataset
data <- read_csv("SeoulBikeData.csv")

# Rename columns based on provided names for better readability and consistency
colnames(data) <- c("Date", "bike_count", "hour", "temp", "humidity", 
                    "wind", "visibility", "dew_pt_temp", "radiation", 
                    "rain", "snow", "seasons", "holiday", "functional")

# Drop columns identified as less relevant (based on prior Lasso selection)
data_selected <- data[, c("bike_count", "hour", "temp", "humidity", 
                          "wind", "visibility", "dew_pt_temp", 
                          "radiation", "rain", "snow")]

# Ensure the target variable is numeric for compatibility
data_selected$bike_count <- as.numeric(data_selected$bike_count)


# Normalize features using z-score normalization (exclude target variable from scaling)
data_normalized <- as.data.frame(scale(data_selected[, -1]))
data_normalized$bike_count <- data_selected$bike_count  # Re-attach target variable

# Set a seed for reproducibility
set.seed(123)

#Linear Regression with Cross-Validation ----
# Define 10-fold cross-validation control
train_control <- trainControl(method = "cv", number = 10)

# Train the Linear Regression model with cross-validation
linear_model_cv <- train(bike_count ~ ., data = data_normalized,
                         method = "lm",
                         trControl = train_control)

# Make predictions and evaluate performance for Linear Regression
linear_cv_predictions <- predict(linear_model_cv, data_normalized)
linear_cv_mse <- mean((linear_cv_predictions - data_normalized$bike_count)^2)
linear_cv_mae <- mean(abs(linear_cv_predictions - data_normalized$bike_count))
linear_cv_correlation <- cor(linear_cv_predictions, data_normalized$bike_count)
cat("Linear Regression with Cross-Validation Metrics:\n")
cat("MSE:", linear_cv_mse, "\nMAE:", linear_cv_mae, "\nCorrelation:", linear_cv_correlation, "\n")

#Multiple Linear Regression with Cross-Validation ----
# Train the Multiple Linear Regression model with 10-fold cross-validation
mlr_cv_model <- train(bike_count ~ ., data = data_normalized,
                      method = "lm",
                      trControl = train_control)

# Make predictions and evaluate performance for Multiple Linear Regression
mlr_cv_predictions <- predict(mlr_cv_model, data_normalized)
mlr_cv_mse <- mean((mlr_cv_predictions - data_normalized$bike_count)^2)
mlr_cv_mae <- mean(abs(mlr_cv_predictions - data_normalized$bike_count))
mlr_cv_correlation <- cor(mlr_cv_predictions, data_normalized$bike_count)
mlr_r_squared <- summary(mlr_cv_model$finalModel)$r.squared
cat("Multiple Linear Regression with Cross-Validation Metrics:\n")
cat("MSE:", mlr_cv_mse, "\nMAE:", mlr_cv_mae, "\nCorrelation:", mlr_cv_correlation, "\nR-squared:", mlr_r_squared, "\n")

#Decision Tree ----
# Train a Decision Tree model
decision_tree_model <- rpart(bike_count ~ ., data = data_normalized)

# Make predictions and evaluate performance for Decision Tree
dt_predictions <- predict(decision_tree_model, data_normalized)
dt_mse <- mean((dt_predictions - data_normalized$bike_count)^2)
dt_mae <- mean(abs(dt_predictions - data_normalized$bike_count))
dt_correlation <- cor(dt_predictions, data_normalized$bike_count)
dt_r_squared <- 1 - (dt_mse / var(data_normalized$bike_count))
cat("Decision Tree Metrics:\n")


cat("MSE:", dt_mse, "\nMAE:", dt_mae, "\nCorrelation:", dt_correlation, "\nR-squared:", dt_r_squared, "\n")

#Neural Network ----
# Train a Neural Network model with a single hidden layer (size = 5)
nn_model <- nnet(bike_count ~ ., data = data_normalized,
                 size = 5, linout = TRUE, trace = FALSE)

# Make predictions and evaluate performance for Neural Network
nn_predictions <- predict(nn_model, data_normalized)
nn_mse <- mean((nn_predictions - data_normalized$bike_count)^2)
nn_mae <- mean(abs(nn_predictions - data_normalized$bike_count))
nn_correlation <- cor(nn_predictions, data_normalized$bike_count)
nn_r_squared <- 1 - (nn_mse / var(data_normalized$bike_count))
cat("Neural Network Metrics:\n")
cat("MSE:", nn_mse, "\nMAE:", nn_mae, "\nCorrelation:", nn_correlation, "\nR-squared:", nn_r_squared, "\n")

#Random Forest ----
# Train a Random Forest model (500 trees)
rf_model <- randomForest(bike_count ~ ., data = data_normalized, importance = TRUE, ntree = 500)

# Make predictions and evaluate performance for Random Forest
rf_predictions <- predict(rf_model, data_normalized)
rf_mse <- mean((rf_predictions - data_normalized$bike_count)^2)
rf_mae <- mean(abs(rf_predictions - data_normalized$bike_count))
rf_correlation <- cor(rf_predictions, data_normalized$bike_count)
rf_r_squared <- 1 - (rf_mse / var(data_normalized$bike_count))
cat("Random Forest Metrics:\n")
cat("MSE:", rf_mse, "\nMAE:", rf_mae, "\nCorrelation:", rf_correlation, "\nR-squared:", rf_r_squared, "\n")

#XGBoost ----
# Prepare data for XGBoost by encoding categorical features and removing unnecessary columns
data <- read.csv("SeoulBikeData.csv")
data$Seasons <- as.factor(data$Seasons)
data$Holiday <- as.factor(data$Holiday)
data$Functioning.Day <- as.factor(data$Functioning.Day)
data <- data %>% select(-Date)

# Define target variable and features
y <- data$Rented.Bike.Count
X <- data %>% select(-Rented.Bike.Count)

# Split into training and testing sets
set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[trainIndex, ]
X_test <- X[-trainIndex, ]
y_train <- y[trainIndex]
y_test <- y[-trainIndex]


# Convert data to DMatrix format for XGBoost
X_train_matrix <- sparse.model.matrix(~ . - 1, data = X_train)
X_test_matrix <- sparse.model.matrix(~ . - 1, data = X_test)
dtrain <- xgb.DMatrix(data = X_train_matrix, label = y_train)
dtest <- xgb.DMatrix(data = X_test_matrix, label = y_test)

# Set XGBoost parameters
params <- list(
  booster = "gbtree",
  objective = "reg:squarederror",
  eta = 0.1,
  max_depth = 6,
  subsample = 0.8,
  colsample_bytree = 0.8
)

# Train the XGBoost model
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100, 
                       watchlist = list(train = dtrain, test = dtest), 
                       early_stopping_rounds = 10, 
                       print_every_n = 10)

# Make predictions and evaluate performance for XGBoost
y_pred_xgb <- predict(xgb_model, dtest)
mse_xgb <- mean((y_test - y_pred_xgb)^2)
r_squared_xgb <- 1 - (sum((y_test - y_pred_xgb)^2) / sum((y_test - mean(y_test))^2))
cat("XGBoost Metrics:\n")
cat("MSE:", mse_xgb, "\nR-squared:", r_squared_xgb, "\n")

# Load ggplot2 for plotting
library(ggplot2)

# -------------------- Plot for Linear Regression --------------------
p1=ggplot(data.frame(Actual = data_normalized$bike_count, Predicted = linear_cv_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("Linear Regression: Actual vs Predicted") +
  xlab("Actual Bike Count") + ylab("Predicted Bike Count")
  print(p1)

# -------------------- Plot for Multiple Linear Regression --------------------
p2=ggplot(data.frame(Actual = data_normalized$bike_count, Predicted = mlr_cv_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("Multiple Linear Regression: Actual vs Predicted") +
  xlab("Actual Bike Count") + ylab("Predicted Bike Count")
  print(p2)

# -------------------- Plot for Decision Tree --------------------
p3=ggplot(data.frame(Actual = data_normalized$bike_count, Predicted = dt_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("Decision Tree: Actual vs Predicted") +
  xlab("Actual Bike Count") + ylab("Predicted Bike Count")
  print(p3)

# -------------------- Plot for Neural Network --------------------
p4=ggplot(data.frame(Actual = data_normalized$bike_count, Predicted = nn_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("Neural Network: Actual vs Predicted") +
  xlab("Actual Bike Count") + ylab("Predicted Bike Count")
  print(p4)

# -------------------- Plot for Random Forest --------------------
p5=ggplot(data.frame(Actual = data_normalized$bike_count, Predicted = rf_predictions), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("Random Forest: Actual vs Predicted") +
  xlab("Actual Bike Count") + ylab("Predicted Bike Count")
  print(p5)

# -------------------- Plot for XGBoost --------------------
p6=ggplot(data.frame(Actual = y_test, Predicted = y_pred_xgb), aes(x = Actual, y = Predicted)) +
  geom_point(color = "blue") +
  geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
  ggtitle("XGBoost: Actual vs Predicted") +
  xlab("Actual Bike Count") + ylab("Predicted Bike Count")
  print(p6)


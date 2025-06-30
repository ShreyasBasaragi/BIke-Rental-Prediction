# Load required libraries
library(shiny)
library(readr)
library(dplyr)

# Load the dataset and prepare the model
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

# Source the Random Forest script
source("randomforest.r")  # This should load the model into an object named rf_model

# Define the UI
ui <- fluidPage(
  titlePanel("Bike Rental Prediction"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("hour", "Hour (0-23):", min = 0, max = 23, value = 12),
      numericInput("temp", "Temperature (°C):", value = 20),
      numericInput("humidity", "Humidity (%):", value = 60),
      numericInput("wind", "Wind Speed (m/s):", value = 0.5),
      numericInput("visibility", "Visibility (10m):", value = 10),
      numericInput("dew_pt_temp", "Dew Point Temperature (°C):", value = 15),
      numericInput("radiation", "Solar Radiation (MJ/m2):", value = 5),
      numericInput("rain", "Rainfall (mm):", value = 0),
      numericInput("snow", "Snowfall (cm):", value = 0),
      actionButton("predict", "Predict")
    ),
    
    mainPanel(
      textOutput("prediction"),
      verbatimTextOutput("debug")  # For debugging messages
    )
  )
)

# Define the server
server <- function(input, output) {
  observeEvent(input$predict, {
    tryCatch({
      # Prepare input data for prediction
      new_data <- data.frame(
        hour = input$hour,
        temp = input$temp,
        humidity = input$humidity,
        wind = input$wind,
        visibility = input$visibility,
        dew_pt_temp = input$dew_pt_temp,
        radiation = input$radiation,
        rain = input$rain,
        snow = input$snow
      )
      
      # Set the column names to match the training data
      colnames(new_data) <- c("hour", "temp", "humidity", "wind", "visibility", 
                              "dew_pt_temp", "radiation", "rain", "snow")
      
      # Normalize the input data using the same means and standard deviations
      means <- colMeans(data_normalized[, -1])
      sds <- apply(data_normalized[, -1], 2, sd)
      
      # Normalize new data using the same means and sds as training
      new_data_normalized <- as.data.frame(scale(new_data, center = means, scale = sds))
      
      # Make prediction
      prediction <- predict(rf_model, new_data_normalized)
      
      # Output the prediction
      output$prediction <- renderText({
        paste("Predicted Number of Rented Bikes:", round(prediction))
      })
      
      # Debugging output
      output$debug <- renderText({
        paste("Input data:", paste(new_data, collapse = ", "), 
              "\nNormalized data:", paste(new_data_normalized, collapse = ", "))
      })
      
    }, error = function(e) {
      # Handle error and display message
      output$prediction <- renderText({
        paste("Error occurred:", e$message)
      })
      output$debug <- renderText({
        "Error in prediction process. Please check your inputs."
      })
    })
  })
}

# Run the app
shinyApp(ui = ui, server = server)

# Weather Prediction and Analysis using Raspberry Pi and Machine Learning

## Overview 

This project is a comprehensive weather prediction system that leverages a Raspberry Pi, various sensors, and machine learning models to forecast local weather conditions in Bangalore. The system predicts future temperatures using a Long Short-Term Memory (LSTM) neural network and detects the probability of rainfall using a Support Vector Machine (SVM).

The project involves real-time data collection, analysis of historical weather patterns, and the application of predictive modeling.

---

## Hardware Components 

* **Central Unit**: Raspberry Pi (Model 4/3B+ recommended)
* **Humidity Sensor**: A sensor like the **DHT11/DHT22** is used for collecting real-time humidity data.
* **Pressure Sensor**: A sensor like the **BMP180/BME280** is used for collecting real-time atmospheric pressure data.

---

## Software and Models 

* **Programming Language**: Python 3
* **Core Libraries**: Pandas, Matplotlib, Seaborn, Scikit-learn, TensorFlow/Keras
* **Temperature Prediction Model**: A **Long Short-Term Memory (LSTM)** recurrent neural network is trained on historical temperature data to forecast future values.
* **Rainfall Detection Model**: A **Support Vector Machine (SVM)** classifier is used to predict the likelihood of rainfall based on various atmospheric features.

---

## Data Acquisition 

This project utilizes a hybrid approach to data collection, combining historical datasets with real-time sensor readings.

### Historical Data

1.  **Temperature Data**: The historical daily average temperature data for **Bangalore (Lat: 12.5, Lon: 77.5)** from the year 2000 to 2023 was acquired from the official **India Meteorological Department (IMD), Pune** data portal at `imdpune.gov.in`. This gridded dataset provides the foundational information for training our LSTM model.

2.  **Humidity and Pressure Data**: To enrich the historical dataset, minimum and maximum daily humidity and pressure readings were acquired using a custom Python script. This script scraped data from a public weather station located near Bangalore, providing a comprehensive atmospheric profile for each day in the historical temperature dataset.

The combined dataset was compiled into the `Bangalore_TEMP-prediction_final.csv` file used for analysis.

---

## Project Workflow and Steps Completed 

### Phase 1: Project Setup

The initial phase involved setting up a Python virtual environment to manage project dependencies and acquiring the necessary historical datasets as described above.

### Phase 2: Exploratory Data Analysis (EDA)

Before training any models, a thorough Exploratory Data Analysis was performed to understand the patterns, trends, and relationships within the weather data. The following steps were completed:

1.  **Data Loading and Inspection**: The `Bangalore_TEMP-prediction_final.csv` dataset was loaded into a pandas DataFrame. An initial inspection was performed to check the data types, column names, and look for any missing values.

2.  **Data Cleaning and Preparation**: The `date` column was converted from a text format to a proper `datetime` object. It was then set as the index of the DataFrame to enable powerful time-series analysis.

3.  **Visualizing Temperature Over Time**: A line chart was generated to plot the daily average temperature over the entire 23-year period. This visualization helped identify long-term trends and the overall cyclical nature of the data.

4.  **Analyzing Seasonality**: To get a clearer view of seasonal patterns, the data was resampled to a monthly average. A new line chart of the monthly average temperature clearly showed the annual rise and fall of temperatures, a critical pattern for the LSTM model to learn.

5.  **Viewing Temperature Distribution**: A histogram was plotted to show the frequency distribution of daily average temperatures. This revealed the most common temperature range in Bangalore and the overall shape of the data distribution.

6.  **Examining Variable Correlations**: Finally, a correlation heatmap was generated to analyze the relationships between different variables (temperature, humidity, pressure). This showed how strongly each variable is related to the others, providing insights into which features might be most predictive.

### Phase 3: Feature Engineering Discussion

A key decision was made regarding the `humidity_min/max` and `pressure_min/max` columns. It was determined that **keeping these columns separate is better than averaging them**. This approach provides the LSTM model with more detailed information about the daily range and volatility of these features, which are powerful predictive indicators that would be lost with a simple average.

---

## Next Steps 

* **Data Preprocessing for LSTM**: Scale the features and create input/output sequences from the time-series data.
* **LSTM Model Training**: Build, compile, and train the LSTM model on the prepared data.
* **Model Evaluation**: Evaluate the model's performance using metrics like Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE).
* **SVM Model Training**: Train the SVM classifier for rainfall detection.
* **Deployment on Raspberry Pi**: Deploy the trained models onto the Raspberry Pi to make predictions using real-time sensor data.

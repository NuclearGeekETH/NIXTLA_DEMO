# Python Time Series Forecasting Tutorial

Welcome to the Python Time Series Forecasting Tutorial using the neuralforecast library powered by [NIXTLA](https://nixtlaverse.nixtla.io/)! This project focuses on providing educational examples about using machine learning techniques for time series prediction in Python. We will specifically explore time series forecasting with Long Short-Term Memory Networks (LSTM) and N-HiTS (Neural Hierarchical Interpolation for Time Series forecast) models.

Our example pulls financial time series data of Tesla and Bitcoin from Yahoo Finance and applies deep learning models to forecast future prices.

## Prerequisites

This tutorial assumes basic familiarity with Python programming, as well as rudimentary knowledge of data analysis and time series. For running the scripts, ensure you have the following:

- Python 3.8 or above
- pip package manager

## Installation

Before running the scripts, you need to install the required libraries. You can install the dependencies by running the following command:

```
pip install -r requirements.txt
```

## Project Structure

The codebase is structured as follows:

- Data Collection: Collect the historical prices of Bitcoin using the Yahoo Finance API (yfinance).
- Data Preparation: Preprocess the data for model ingestion.
- Model Training: Train LSTM and N-HiTS models on the historical data.
- Prediction: Predict future values using trained models.
- Validation: Split the dataset into training and validation subsets and evaluate the models.
- Visualization: Visualize both the actual and predicted values using plotly.

## Usage

1. Initialize your Python environment and make sure all dependencies are installed.
2. Run the script: Execute the provided Python file. The script will fetch data, perform training, and visualize the predictions.
3. Interpret the output: The console will display mean absolute errors for each model, and a web-based interactive plot will show the actual vs. predicted prices.

## Key Features

The script includes:

- Implementation of LSTM and N-HiTS models for forecasting.
- Visualization of results with interactive charts.
- Calculation and display of the Mean Absolute Error (MAE) to evaluate model performance.
- Examples of how to preprocess financial time series data for neural network models.

## Contributing

Feel free to fork this repository and submit pull requests. You can also submit issues if you find bugs or have suggestions.

## Acknowledgments
- Special thanks to user [GUSLOVESMATH](https://www.kaggle.com/guslovesmath) at Kaggle for the inspiration for this repo. Check his [Kaggle Notebook](https://www.kaggle.com/code/guslovesmath/amazing-neuralforecast-nvda-forecasting). 
- Thanks to the creators of the neuralforecast library for providing extensive tools for time series analysis.
- This tutorial utilizes data provided by Yahoo Finance.
- Special thanks to contributors and maintainers of the pandas, numpy, plotly, and yfinance libraries.

Feel free to dive into the code, experiment with different models, and explore how changes in parameters affect the forecasting accuracy! If you have questions or need further assistance, consider reaching out to the machine learning and data science communities on platforms like Stack Overflow or Reddit and using AI tools to develop your code more.
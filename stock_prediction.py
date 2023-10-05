# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import keras
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
import pickle
import statsmodels.api as sm
import csv

from matplotlib.dates import DateFormatter, date2num
from mpl_finance import candlestick_ohlc
from pandas.io.xml import preprocess_data
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, RNN, GRU
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from statsmodels.tsa.arima.model import ARIMA as arima_model

from parameters import prediction_days, lstm_units, num_epochs, batch_size, dropout_rate
from parameters import version
from parameters import tick
from parameters import train_end
from parameters import train_start
from parameters import data_source
from parameters import train_test_ratio
from parameters import split_method
from parameters import scale_features
from parameters import price_value
from parameters import test_start
from parameters import test_end


# from parameters import lstm_units
# from parameters import dropout_rate
# from parameters import num_epochs
# from parameters import batch_size


# ------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
# ------------------------------------------------------------------------------


# data = yf.download(tick, start=train_start, end=train_end)

# Function to load or fetch data
def load_data(data_source, tick, train_start, train_end, train_test_ratio, split_method, scale_features):
    # Check if the directory exists; if not, create it
    if not os.path.exists(data_source):
        os.makedirs(data_source)

    # Construct the file path for data saving/loading
    version = 1  # You might want to set a specific version number
    file_path = os.path.join(data_source, f"{tick}_v{version}.txt")

    # Check if saved data exists; if yes, load it; if not, fetch from Yahoo Finance
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print("DATA loaded successfully.")
    else:
        # Load data from Yahoo Finance
        data = yf.download(tick, start=train_start, end=train_end)

        # Handle NaN values using forward filling
        data = data.ffill().dropna()

        # Save the fetched data
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print("Data saved successfully.")

    # Split data into train and test sets based on the value {split_method}
    if split_method == 'date':
        # If split method is 'date', calculate the date for the end of the training time
        train_end_date = train_start + pd.DateOffset(days=int((train_end - train_start).days * train_test_ratio))
        # Extract the data for the training period (from TRAIN_START to train_end_date)
        train_d = data[train_start:train_end_date]
        # Extract the data for the testing period (from train_end_date to TRAIN_END)
        test_d = data[train_end_date:train_end]
    elif split_method == 'random':
        # If split method is 'random', use train_test_split to split the data randomly
        # The train_size parameter specifies the ratio of training data
        # The shuffle parameter is set too False to maintain order of the data

        train_d, test_d = train_test_split(data, train_size=train_test_ratio, shuffle=False)
    else:
        # If an invalid split method is specified, raise a ValueError
        raise ValueError("Invalid split method specified.")

    # Scale feature columns if specified
    scalers = {}
    if scale_features:
        feature_columns = train_d.columns
        scaler = StandardScaler()  # Instantiate StandardScaler object, removing the mean, scaling to unit variance
        train_d[feature_columns] = scaler.fit_transform(train_d[feature_columns])  # Fit and transform training data
        test_d[feature_columns] = scaler.transform(test_d[feature_columns])  # Transform test data using the same scaler
        scalers['feature'] = scaler  # Store the scaler in the dictionary

    if not scalers:
        scalers['feature'] = None
        scalers['target'] = None

    return train_d, test_d, scalers


# Load data using the load_data function
train_d, test_d, scalers = load_data(data_source, tick, train_start, train_end, train_test_ratio, split_method,
                                     scale_features)
print("Train Data: ")
print(train_d)
print("Test Data: ")
print(test_d)
print("Scalers: ")
print(scalers)

# TASK B.2 END

# For more details:
# https://pandas.pydata.org/pandas-docs/stable/user_guide/dsintro.html
# ------------------------------------------------------------------------------
# Prepare Data
## To do:
# 1) Check if data has been prepared before. 
# If so, load the saved data
# If not, save the data into a directory
# 2) Use a different price value eg. mid-point of Open & Close
# 3) Change the Prediction days
# ------------------------------------------------------------------------------

# Initialize MinMaxScaler for feature scaling
scaler = MinMaxScaler(feature_range=(0, 1))

# Scale the price data
scaled_data = scaler.fit_transform(train_d[price_value].values.reshape(-1, 1))

# Flatten the scaled data from 2D to 1D
scaled_data = scaled_data[:, 0]

# Prepare training data and labels
x_train = []
y_train = []

# Use a sliding window approach to create training samples
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x])
    y_train.append(scaled_data[x])

# Convert lists to NumPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

# Reshape x_train to be a 3D array (samples, time steps, features)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# ------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before.
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
# ------------------------------------------------------------------------------
#
# Function to build a deep learning model with GRU layers
def build_deep_learning_model_with_gru(layers, layer_sizes, layer_names, input_shape):
    ## Check if the input lists have the same length, optional task
    if len(layers) != len(layer_sizes) or len(layers) != len(layer_names):
        raise ValueError("The number of layers, layer_sizes, and layer_names must match.")

# Create a Sequential model
    model = Sequential()

# Loop through the specific layers
    for i in range(len(layers)):
        layer_name = layer_names[i]
        layer_size = layer_sizes[i]

        if layers[i] == "GRU":
# Add a GRU layer
            print(f"Adding GRU layer with units={layer_size}, return_sequences=True, input_shape={input_shape}, name={layer_name}")
            model.add(GRU(units=layer_size, return_sequences=True, input_shape=input_shape, name=layer_name))
        elif layers[i] == "Dropout":
# Add a Dropout layer
            print(f"Adding Dropout layer with rate={layer_size}, name={layer_name}")
            model.add(Dropout(layer_size, name=layer_name))
        elif layers[i] == "Dense":
# Add a Dense (fully connected) layer
            print(f"Adding Dense layer with units={layer_size}, name={layer_name}")
            model.add(Dense(units=layer_size, name=layer_name))

    return model

# Function to predict stock prices using GRU model
# def predict_stock_prices_with_gru(tick, test_start, test_end, prediction_days=1, model=None):
#     # Download historical stock data
#     data = yf.download(tick, start=test_start, end=test_end, progress=False)
#
#     # Prepare the data using Min-Max scaling
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#
# # Prepare the input data for the GRU model
#     x_test = []
#     for x in range(prediction_days, len(scaled_data)):
#         x_test.append(scaled_data[x - prediction_days:x, 0])
#
#     x_test = np.array(x_test)
#     x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
#
#     # Build the GRU model if not provided
#     if model is None:
#         print("Building GRU model...")
#         model = build_deep_learning_model_with_gru(layers=["GRU", "Dropout", "GRU", "Dropout", "GRU", "Dropout", "Dense"],
#                                                   layer_sizes=[50, 0.2, 50, 0.2, 50, 0.2, 1],
#                                                   layer_names=["gru1", "dropout1", "gru2", "dropout2", "gru3", "dropout3",
#                                                                "dense1"],
#                                                   input_shape=(x_test.shape[1], 1))
#         # Compile the model
#         print("Compiling the model...")
#         model.compile(optimizer='adam', loss='mean_squared_error')
#
#         # Load pre-trained weights if available
#         model_weights_path = f"{tick}_model_weights_gru.h5"
#         if os.path.exists(model_weights_path):
#             print("Loading pre-trained weights...")
#             model.load_weights(model_weights_path)
#
#     # Predict the stock prices
#     print("Predicting stock prices with GRU...")
#     predicted_prices = model.predict(x_test)
#     predicted_prices = predicted_prices.reshape(-1, 1)  # Flatten and reshape
#
#     predicted_prices = scaler.inverse_transform(predicted_prices)
#
#     # Predict the next day's closing price
#     real_data = [scaled_data[len(scaled_data) - prediction_days:, 0]]
#     real_data = np.array(real_data)
#     real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))
#
#     next_day_prediction = model.predict(real_data)
#     next_day_prediction = next_day_prediction.reshape(-1, 1)  # Flatten and reshape
#
#     next_day_prediction = scaler.inverse_transform(next_day_prediction)
#
#
#     return predicted_prices, next_day_prediction[0][0]
#
# predicted_prices, next_day_prediction = predict_stock_prices_with_gru(tick, train_start, train_end)

## GRU ^^^^^^^^^

# Function to build a deep learning model
def build_deep_learning_model(layers, layer_sizes, layer_names, input_shape):
    # Check if the input lists have the same length
    if len(layers) != len(layer_sizes) or len(layers) != len(layer_names):
        raise ValueError("The number of layers, layer_sizes, and layer_names must match.")

    # Create a Sequential model
    model = Sequential()

    # Loop through the specified layers
    for i in range(len(layers)):
        layer_name = layer_names[i]
        layer_size = layer_sizes[i]

        if layers[i] == "LSTM":
            # Add an LSTM layer
            print(
                f"Adding LSTM layer with units={layer_size}, return_sequences=True, input_shape={input_shape}, name={layer_name}")
            model.add(LSTM(units=layer_size, return_sequences=True, input_shape=input_shape, name=layer_name))
        elif layers[i] == "Dropout":
            # Add a Dropout layer
            print(f"Adding Dropout layer with rate={layer_size}, name={layer_name}")
            model.add(Dropout(layer_size, name=layer_name))
        elif layers[i] == "Dense":
            # Add a Dense (fully connected) layer
            print(f"Adding Dense layer with units={layer_size}, name={layer_name}")
            model.add(Dense(units=layer_size, name=layer_name))

    return model


# Function to predict stock prices
def predict_stock_prices(tick, test_start, test_end, prediction_days=1, model=None):
    # Download historical stock data
    data = yf.download(tick, start=test_start, end=test_end, progress=False)

    # Prepare the data using Min-Max scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))

    # Prepare the input data for the LSTM model
    x_test = []
    for x in range(prediction_days, len(scaled_data)):
        x_test.append(scaled_data[x - prediction_days:x, 0])

    x_test = np.array(x_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

    # Build the LSTM model if not provided
    if model is None:
        print("Building LSTM model...")
        model = build_deep_learning_model(layers=["LSTM", "Dropout", "LSTM", "Dropout", "LSTM", "Dropout", "Dense"],
                                          layer_sizes=[50, 0.2, 50, 0.2, 50, 0.2, 1],
                                          layer_names=["lstm1", "dropout1", "lstm2", "dropout2", "lstm3", "dropout3",
                                                       "dense1"],
                                          input_shape=(x_test.shape[1], 1))
        # Compile the model
        print("Compiling the model...")
        model.compile(optimizer='adam', loss='mean_squared_error')

        # Load pre-trained weights if available
        model_weights_path = f"{tick}_model_weights.h5"
        if os.path.exists(model_weights_path):
            print("Loading pre-trained weights...")
            model.load_weights(model_weights_path)

    # Predict the stock prices
    print("Predicting stock prices...")
    predicted_prices = model.predict(x_test)
    predicted_prices = predicted_prices.reshape(-1, 1)  # Flatten and reshape

    predicted_prices = scaler.inverse_transform(predicted_prices)

    # Predict the next day's closing price
    real_data = [scaled_data[len(scaled_data) - prediction_days:, 0]]
    real_data = np.array(real_data)
    real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

    next_day_prediction = model.predict(real_data)
    next_day_prediction = next_day_prediction.reshape(-1, 1)  # Flatten and reshape

    next_day_prediction = scaler.inverse_transform(next_day_prediction)

    return predicted_prices, next_day_prediction[0][0]


# Usage
predicted_prices, next_day_prediction = predict_stock_prices(tick, train_start, train_end)
#
print(f"Predicted Prices for the next {prediction_days} days: {predicted_prices}")
print(f"Predicted Price for the next day: {next_day_prediction}")

# Create a Sequential model
model = Sequential()

# Add the first LSTM layer with specified units, return_sequences, and input_shape
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))

# Add a Dropout layer to prevent overfitting
model.add(Dropout(0.2))

# Add a second LSTM layer
model.add(LSTM(units=50, return_sequences=True))

# Add another Dropout layer
model.add(Dropout(0.2))

# Add a third LSTM layer
model.add(LSTM(units=50))

# Add a Dropout layer
model.add(Dropout(0.2))

# Add a Dense (fully connected) layer for prediction
model.add(Dense(units=1))

# Compile the model with optimizer and loss function
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model with training data (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)


# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

## START OF B.6 - Task 6 - Machine Learning 3 ##

def build_arima_model(train_d):
    # Fit an ARIMA model to the training data
    model = sm.tsa.ARIMA(train_d, order=(1, 1, 1))
    arima_result = model.fit()
    return arima_result

def predict_arima(arima_model, test_d):
    # Make predictions using the ARIMA model
    arima_predictions = arima_model.forecast(steps=len(test_d))
    return arima_predictions


# Load and preprocess the data
train_d, test_d, _ = load_data(data_source, tick, train_start, train_end, train_test_ratio, split_method,
                                     scale_features)

# Define the architecture LSTM model
layers = ["LSTM", "Dropout", "LSTM", "Dropout", "LSTM", "Dropout", "Dense"]
layer_sizes = [50, 0.2, 50, 0.2, 50, 0.2, 1]
layer_names = ["lstm1", "dropout1", "lstm2", "dropout2", "lstm3", "dropout3", "dense1"]
input_shape = (x_train.shape[1], x_train.shape[2])

# Build and compile the LSTM model
lstm_model = build_deep_learning_model(layers, layer_sizes, layer_names, input_shape)
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train the LSTM model
lstm_model.fit(x_train, y_train, epochs=25, batch_size=32)

# # Build and train the ARIMA model
arima_model = build_arima_model(train_d['Close'])
#
# # Predict using the LSTM model
lstm_predicted_prices, lstm_next_day_prediction = predict_stock_prices(tick, test_start, test_end, prediction_days,
                                                                       lstm_model)
# # Predict using the ARIMA model
arima_predicted_prices = predict_arima(arima_model, test_d['Close'])
#
# Trim lstm_predicted_prices to match the length of arima_predicted_prices
lstm_predicted_prices = lstm_predicted_prices[:len(arima_predicted_prices)]

# Assuming lstm_predicted_prices is the problematic array
lstm_predicted_prices = lstm_predicted_prices.ravel()

print("Length of LSTM Predictions:", len(lstm_predicted_prices))
print("Length of ARIMA Predictions:", len(arima_predicted_prices))

# # Combine predictions from both models (simple average)
weight_lstm = 0.5
weight_arima = 0.5
ensemble_predictions = (weight_lstm * lstm_predicted_prices) + (weight_arima * arima_predicted_prices)
#
# # Print ensemble predictions
print("Ensemble Predictions: ", ensemble_predictions)

# Calculate MSE and MAE for ARIMA
mse_arima = mean_squared_error(test_d['Close'], arima_predicted_prices)
mae_arima = mean_absolute_error(test_d['Close'], arima_predicted_prices)

# Calculate MSE and MAE for LSTM
mse_lstm = mean_squared_error(test_d['Close'], lstm_predicted_prices)
mae_lstm = mean_absolute_error(test_d['Close'], lstm_predicted_prices)

# Calculate MSE and MAE for the ensemble
mse_ensemble = mean_squared_error(test_d['Close'], ensemble_predictions)
mae_ensemble = mean_absolute_error(test_d['Close'], ensemble_predictions)

print("MSE for ARIMA:", mse_arima)
print("MAE for ARIMA:", mae_arima)

print("MSE for LSTM:", mse_lstm)
print("MAE for LSTM:", mae_lstm)

print("MSE for Ensemble:", mse_ensemble)
print("MAE for Ensemble:", mae_ensemble)

# Integrating a RandomForrest

# Prepare the data for RandomForest model
test_d_array = test_d.to_numpy()
# Reshape the data to 2D format
x_train_2d = x_train.reshape(x_train.shape[0], -1)
x_test_2d = test_d_array.reshape(test_d_array.shape[0], -1)

# # Ensure that x_test_2d has the same number of features as x_train_2d
# num_features_train = x_train_2d.shape[1]
# num_features_test = x_test_2d.shape[1]

# Ensure that x_test_2d has the same number of features as x_train_2d
if x_test_2d.shape[1] < x_train_2d.shape[1]:
    num_features_to_add = x_train_2d.shape[1] - x_test_2d.shape[1]
    extra_features = np.zeros((x_test_2d.shape[0], num_features_to_add))
    x_test_2d = np.hstack((x_test_2d, extra_features))

# # If the number of features doesn't match
# if num_features_test < num_features_train:
#     # Add extra features to x_test_2d, e.g., zeros
#     num_features_to_add = num_features_train - num_features_test
#     extra_features = np.zeros((x_test_2d.shape[0], num_features_to_add))
#     x_test_2d = np.hstack((x_test_2d, extra_features))
# elif num_features_test > num_features_train:
#     # Remove extra features from x_test_2d
#     x_test_2d = x_test_2d[:, :num_features_train]

#
# # The below commented out code runs through all the hyperparameters to determine which
# # returns the best result
# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [10, 15, 20],
#     'min_samples_split': [2, 5, 10],
#     'min_samples_leaf': [1, 2, 4],
#     'max_features': [0.2, 0.5, 0.8]
# }
#
# grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3, n_jobs=-1)
# grid_search.fit(x_train_2d, y_train)
#
# best_rf_model = grid_search.best_estimator_
# best_rf_predicted_prices = best_rf_model.predict(x_test_2d)
#
# # Calculate MSE and MAE for the RandomForest model
# mse_rf = mean_squared_error(test_d['Close'], best_rf_predicted_prices)
# mae_rf = mean_absolute_error(test_d['Close'], best_rf_predicted_prices)
#
# print("MSE for RandomForest:", mse_rf)
# print("MAE for RandomForest:", mae_rf)


# Create the RandomForestRegressor with custom parameters
rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=1,
    max_features=0.2,
    random_state=42
)
rf_model.fit(x_train_2d, y_train)

# Predict using the RandomForest model
rf_predicted_prices = rf_model.predict(x_test_2d)

# Combine predictions from all models
weight_rf = 0.2
ensemble_predictions = (weight_lstm * lstm_predicted_prices) + (weight_arima * arima_predicted_prices) + (weight_rf * rf_predicted_prices)

# Calculate MSE and MAE for the RandomForest model
mse_rf = mean_squared_error(test_d['Close'], rf_predicted_prices)
mae_rf = mean_absolute_error(test_d['Close'], rf_predicted_prices)

print("MSE for RandomForest:", mse_rf)
print("MAE for RandomForest:", mae_rf)
print("Ensemble predictions with RF: ", ensemble_predictions)

with open('ensemble_predictions_GRIDSEARCH.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Prediction'])  # Optional header row
    writer.writerows(zip(ensemble_predictions))

## END OF B.6 ##


# ------------------------------------------------------------------------------
# Test the model accuracy on existing data
# ------------------------------------------------------------------------------
# Load the test data
test_data = yf.download(tick, start=test_start, end=test_end, progress=False)

# Remove the first row to fix a bug
test_data = test_data[1:]

# Get the actual prices from the test data
actual_prices = test_data[price_value].values

# Combine the training and test data for model inputs
total_dataset = pd.concat((train_d[price_value], test_data[price_value]), axis=0)

# Get the model inputs by considering data from both training and test periods
model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values

# Reshape the model inputs
model_inputs = model_inputs.reshape(-1, 1)

# Normalize the model inputs using the same scaler as before
model_inputs = scaler.transform(model_inputs)

# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [train_start, train_end],
# but there may be a lower/higher price during the test period 
# [test_start, test_end]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

# ------------------------------------------------------------------------------
# Make predictions on test data
# ------------------------------------------------------------------------------
x_test = []
for x in range(prediction_days, len(model_inputs)):
    x_test.append(model_inputs[x - prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# Make predictions using the model
predicted_prices = model.predict(x_test)

# Inverse transform the predicted prices to the original scale
predicted_prices = scaler.inverse_transform(predicted_prices)
# ------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
# ------------------------------------------------------------------------------
## START OF B.3 candlestick and boxplot ##

# Create the data directory if it doesn't exist
data_directory = "graphs_data"
os.makedirs(data_directory, exist_ok=True)

# Create the output directory based on parameters
output_directory = os.path.join(data_directory, f"{tick}_{version}_{split_method}_{test_start}_{test_end}")
os.makedirs(output_directory, exist_ok=True)

# Convert the index of the DataFrame to Python datetime objects
test_d.index = test_d.index.to_pydatetime()

# Create candlestick bars
ohlc_data = []

# Iterate through each row in the test_d DataFrame
for date, row in test_d.iterrows():
    ohlc = [date2num(date), row['Open'], row['High'], row['Low'], row['Close']]
    ohlc_data.append(ohlc)

# Create a subplot with a specified figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Generate the candlestick chart using the ohlc_data
candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='g', colordown='r')

# Format the x-axis labels with the year-month-day format
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax.set_title(f'Candlestick Chart for {tick}')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
ax.plot(test_d.index, test_d['Close'], label='Closing Price', color='black')
ax.plot(test_d.index, test_d['Open'], label='Opening Price', color='blue')
ax.legend()
ax.grid()
plt.tight_layout()
plt.show()

# Save the candlestick chart as an image file
candlestick_filename = os.path.join(output_directory, f'candlestick_chart.png')
fig.savefig(candlestick_filename)


######################
# END OF CANDLESTICK #
######################

####################
# START OF BOXPLOT #
####################
# Boxplot graph, using test data:
def plot_boxplot_chart(test_d, window_size=2):
    # Calculate the number of moving windows
    num_windows = len(test_d) - window_size + 1

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize lists to store data for each window
    window_data = []

    # Loop through the moving windows
    for i in range(num_windows):
        # Extract data for the current window
        window = test_d[i:i + window_size]
        window_close = window['Close']

        # Append window data to the list
        window_data.append(window_close)

    # Create a boxplot chart using the window data
    ax.boxplot(window_data)

    # Set labels and title
    ax.set_xticklabels([str(i) for i in range(1, num_windows + 1)])
    ax.set_xlabel('Moving Window')
    ax.set_ylabel('Closing Price')
    ax.set_title(f'Boxplot Graph for {tick}')

    # Ensure the chart layout is tight
    plt.tight_layout()

    # Show the boxplot chart
    plt.show()

    # Save the boxplot chart as an image file
    boxplot_filename = os.path.join(output_directory, f'boxplot_chart.png')
    fig.savefig(boxplot_filename)


# Call the function to plot the boxplot chart
plot_boxplot_chart(test_d)

##################
# END OF BOXPLOT #
##################


# Original graph, with line tick plotting:
plt.plot(actual_prices, color="black", label=f"Actual {tick} Price")
plt.plot(predicted_prices, color="red", label=f"Predicted {tick} Price")
plt.title(f"{tick} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{tick} Share Price")
plt.legend()
original_graph_filename = os.path.join(output_directory, f'original_graph.png')
plt.savefig(original_graph_filename)
plt.show()


# ------------------------------------------------------------------------------
# Predict next day
# ------------------------------------------------------------------------------
# TASK B.5 SOLVING MORE ADVANCED prediction problems. Including multivariate prediction and multistep prediction.

# Define a function for multistep prediction using a trained model
def multistep_prediction(train_d, k):
    # Extract the 'Close' column values from the training data as a Numpy array
    data = train_d[['Close']].values

    # Scale the 'Close' values to a range between 0 and 1 using Min-Max scaling
    data_scaled = scaler.fit_transform(data)

    X, y = [], []

    # Create sequences of length k as input (X) and the next value as the target (y)
    for i in range(len(data_scaled) - k):
        X.append(data_scaled[i:i + k])
        y.append(data_scaled[i + k])

    X, y = np.array(X), np.array(y)

    # Predict k days into the future
    predictions = []
    current_sequence = X[-1]  # Start prediction from the end of the training data

    for i in range(k):
        # Predict the next value using the model
        prediction = model.predict(current_sequence.reshape(1, k, 1))

        # Inverse transform the scaled prediction to the original scale
        prediction = scaler.inverse_transform(prediction)[0][0]

        # Append the prediction to the list of predictions
        predictions.append(prediction)

        # Update the current sequence by removing the first element and adding the prediction
        current_sequence = np.append(current_sequence[1:], prediction)

    return predictions


# Define a function for multistep prediction using linear regression
def multivariate_prediction(train_d, k):
    # Define the list of features to be used for the prediction
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

    # Extract the feature values (X) and target values (y) from the training data
    X = train_d[features].values
    y = train_d['Close'].values

    # Create a Linear Regression model
    multivariate_model = LinearRegression()

    # Fit the model to the training data, which means it learns the relationship between features and target
    multivariate_model.fit(X, y)

    # Predict the closing price k days into the future by using the last available data point
    multi_prediction = multivariate_model.predict(X[-k].reshape(1, -1))

    # Return the predicted closing price for the specified day in the future
    return multi_prediction[0]


# Example usage:
k = 5  # Specify the number of days into the future for multistep prediction

# Call the multistep_prediction and multivariate_prediction function.
multistep_predictions = multistep_prediction(train_d, k)
multivariate_prediction = multivariate_prediction(train_d, k)

print("Multistep Predictions:", multistep_predictions)
print(f"Simple Multivariate Prediction for Day {k}:", multivariate_prediction)
## END OF B.5 ##


# comments for old prediction model
# Select the last 'prediction_days' elements from 'model_inputs' for prediction
real_data = [model_inputs[len(model_inputs) - prediction_days:, 0]]
# Convert the selected data into a Numpy array
real_data = np.array(real_data)
# Reshape the Numpy array to match the expected input shape of the model
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

# Use the trained model to make a prediction on the real-data
prediction = model.predict(real_data)
# Inverse transform the scaled prediction to the original scale
prediction = scaler.inverse_transform(prediction)
print(f"Prediction: {prediction}")

# A few concluding remarks here:
# 1. The predictor is quite bad, especially if you look at the next day 
# prediction, it missed the actual price by about 10%-13%
# Can you find the reason?
# 2. The code base at
# https://github.com/x4nth055/pythoncode-tutorials/tree/master/machine-learning/stock-prediction
# gives a much better prediction. Even though on the surface, it didn't seem 
# to be a big difference (both use Stacked LSTM)
# Again, can you explain it?
# A more advanced and quite different technique use CNN to analyse the images
# of the stock price changes to detect some patterns with the trend of
# the stock price:
# https://github.com/jason887/Using-Deep-Learning-Neural-Networks-and-Candlestick-Chart-Representation-to-Predict-Stock-Market
# Can you combine these different techniques for a better prediction??


# Need to install the following:
# pip install numpy
# pip install matplotlib
# pip install pandas
# pip install tensorflow
# pip install scikit-learn
# pip install pandas-datareader
# pip install yfinance

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
import os
import pickle

from matplotlib.dates import DateFormatter, date2num
from mpl_finance import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer, RNN, GRU
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

from parameters import prediction_days
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

scaler = MinMaxScaler(feature_range=(0, 1))
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(train_d[price_value].values.reshape(-1, 1))
# Flatten and normalise the data
# First, we reshape a 1D array(n) to 2D array(n,1)
# We have to do that because sklearn.preprocessing.fit_transform()
# requires a 2D array
# Here n == len(scaled_data)
# Then, we scale the whole array to the range (0,1)
# The parameter -1 allows (np.)reshape to figure out the array size n automatically 
# values.reshape(-1, 1) 
# https://stackoverflow.com/questions/18691084/what-does-1-mean-in-numpy-reshape'
# When reshaping an array, the new shape must contain the same number of elements 
# as the old shape, meaning the products of the two shapes' dimensions must be equal. 
# When using a -1, the dimension corresponding to the -1 will be the product of 
# the dimensions of the original array divided by the product of the dimensions 
# given to reshape so as to maintain the same number of elements.


# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:, 0]  # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - prediction_days
# and q = prediction_days; while y_train is a 1D array(p)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))


# We now reshape x_train into a 3D array(p, q, 1); Note that x_train
# is an array of p inputs with each input being a 2D array 

# ------------------------------------------------------------------------------
# Build the Model
## TO DO:
# 1) Check if data has been built before.
# If so, load the saved data
# If not, save the data into a directory
# 2) Change the model to increase accuracy?
# ------------------------------------------------------------------------------
#
## Function to build a deep learning model with GRU layers
# def build_deep_learning_model_with_gru(layers, layer_sizes, layer_names, input_shape):
#     ## Check if the input lists have the same length, optional task
#     if len(layers) != len(layer_sizes) or len(layers) != len(layer_names):
#         raise ValueError("The number of layers, layer_sizes, and layer_names must match.")
#
## Create a Sequential model
#     model = Sequential()
#
## Loop through the specific layers
#     for i in range(len(layers)):
#         layer_name = layer_names[i]
#         layer_size = layer_sizes[i]
#
#         if layers[i] == "GRU":
## Add a GRU layer
#             print(f"Adding GRU layer with units={layer_size}, return_sequences=True, input_shape={input_shape}, name={layer_name}")
#             model.add(GRU(units=layer_size, return_sequences=True, input_shape=input_shape, name=layer_name))
#         elif layers[i] == "Dropout":
## Add a Dropout layer
#             print(f"Adding Dropout layer with rate={layer_size}, name={layer_name}")
#             model.add(Dropout(layer_size, name=layer_name))
#         elif layers[i] == "Dense":
## Add a Dense (fully connected) layer
#             print(f"Adding Dense layer with units={layer_size}, name={layer_name}")
#             model.add(Dense(units=layer_size, name=layer_name))
#
#     return model
#
## Function to predict stock prices using GRU model
# def predict_stock_prices_with_gru(tick, test_start, test_end, prediction_days=1, model=None):
#     # Download historical stock data
#     data = yf.download(tick, start=test_start, end=test_end, progress=False)
#
#     # Prepare the data using Min-Max scaling
#     scaler = MinMaxScaler(feature_range=(0, 1))
#     scaled_data = scaler.fit_transform(data['Close'].values.reshape(-1, 1))
#
## Prepare the input data for the GRU model
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
#     return predicted_prices, next_day_prediction[0][0]
#
# predicted_prices, next_day_prediction = predict_stock_prices_with_gru(tick, train_start, train_end)

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
            # Add an LSTM layerr
            print(
                f"Adding LSTM layer with units={layer_size}, return_sequences=True, input_shape={input_shape}, name={layer_name}")
            model.add(LSTM(units=layer_size, return_sequences=True, input_shape=input_shape, name=layer_name))
        elif layers[i] == "Dropout":
            # Add a Droupout layer
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

model = Sequential()  # Basic neural network
# See: https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
# for some useful examples

model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
# This is our first hidden layer which also spcifies an input layer. 
# That's why we specify the input shape for this layer; 
# i.e. the format of each training example
# The above would be equivalent to the following two lines of code:
# model.add(InputLayer(input_shape=(x_train.shape[1], 1)))
# model.add(LSTM(units=50, return_sequences=True))
# For some advances explanation of return_sequences:
# https://machinelearningmastery.com/return-sequences-and-return-states-for-lstms-in-keras/
# https://www.dlology.com/blog/how-to-use-return_state-or-return_sequences-in-keras/
# As explained there, for a stacked LSTM, you must set return_sequences=True 
# when stacking LSTM layers so that the next LSTM layer has a 
# three-dimensional sequence input. 

# Finally, units specifies the number of nodes in this layer.
# This is one of the parameters you want to play with to see what number
# of units will give you better prediction quality (for your problem)

model.add(Dropout(0.2))
# The Dropout layer randomly sets input units to 0 with a frequency of 
# rate (= 0.2 above) at each step during training time, which helps 
# prevent overfitting (one of the major problems of ML). 

model.add(LSTM(units=50, return_sequences=True))
# More on Stacked LSTM:
# https://machinelearningmastery.com/stacked-long-short-term-memory-networks/

model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))
# Prediction of the next closing value of the stock price

# We compile the model by specify the parameters for the model
# See lecture Week 6 (COS30018)
model.compile(optimizer='adam', loss='mean_squared_error')
# The optimizer and loss are two important parameters when building an 
# ANN model. Choosing a different optimizer/loss can affect the prediction
# quality significantly. You should try other settings to learn; e.g.

# optimizer='rmsprop'/'sgd'/'adadelta'/...
# loss='mean_absolute_error'/'huber_loss'/'cosine_similarity'/...

# Now we are going to train this model with our training data 
# (x_train, y_train)
model.fit(x_train, y_train, epochs=25, batch_size=32)
# Other parameters to consider: How many rounds(epochs) are we going to 
# train our model? Typically, the more the better, but be careful about
# overfitting!
# What about batch_size? Well, again, please refer to 
# Lecture Week 6 (COS30018): If you update your model for each and every 
# input sample, then there are potentially 2 issues: 1. If you training 
# data is very big (billions of input samples) then it will take VERY long;
# 2. Each and every input can immediately makes changes to your model
# (a souce of overfitting). Thus, we do this in batches: We'll look at
# the aggreated errors/losses from a batch of, say, 32 input samples
# and update our model based on this aggregated loss.

# TO DO:
# Save the model and reload it
# Sometimes, it takes a lot of effort to train your model (again, look at
# a training data with billions of input samples). Thus, after spending so 
# much computing power to train your model, you may want to save it so that
# in the future, when you want to make the prediction, you only need to load
# your pre-trained model and run it on the new input for which the prediction
# need to be made.

# ------------------------------------------------------------------------------
# Test the model accuracy on existing data
# ------------------------------------------------------------------------------
# Load the test data

test_data = yf.download(tick, start=test_start, end=test_end, progress=False)

# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[price_value].values

total_dataset = pd.concat((train_d[price_value], test_data[price_value]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - prediction_days:].values
# We need to do the above because to predict the closing price of the first
# prediction_days of the test period [test_start, test_end], we'll need the
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

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
# TO DO: Explain the above 5 lines

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Clearly, as we transform our data into the normalized range (0,1),
# we now need to reverse this transformation 
# ------------------------------------------------------------------------------
# Plot the test predictions
## To do:
# 1) Candle stick charts
# 2) Chart showing High & Lows of the day
# 3) Show chart of next few days (predicted)
# ------------------------------------------------------------------------------
## START OF B.3 candlestick and boxplot ##

data_directory = "graphs_data"
os.makedirs(data_directory, exist_ok=True)

output_directory = os.path.join(data_directory, f"{tick}_{version}_{split_method}_{test_start}_{test_end}")
os.makedirs(output_directory, exist_ok=True)

##############################
# Start of Candlestick Graph #
##############################
# Convert the index of the DataFrame to Python datetime objects
test_d.index = test_d.index.to_pydatetime()

# Create candlestick bars
# Create a list to hold the OHLC (Open, High, Low, Close) data for the candlestick chart
ohlc_data = []

# Iterate through each row in the test_d DataFrame
for date, row in test_d.iterrows():
    # Convert the data to a numerical format required by candlestick_ohlc function
    # Extract the OHLC values from the row, Create an OHLC tuple and add it to the ohlc_data list
    ohlc = [date2num(date), row['Open'], row['High'], row['Low'], row['Close']]
    ohlc_data.append(ohlc)

# Create a subplot with a specified figure size
fig, ax = plt.subplots(figsize=(10, 6))

# Generate the candlestick chart using the ohlc_data
candlestick_ohlc(ax, ohlc_data, width=0.6, colorup='g', colordown='r')

# Plot the candlestick chart
# Format the x-axis labels with the year-month-day format
ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
# Set the title, x-axis label, and y-axis label
ax.set_title(f'Candlestick Chart for {tick}')
ax.set_xlabel('Date')
ax.set_ylabel('Price')
# Plot the actual closing and opening prices on th chart
ax.plot(test_d.index, test_d['Close'], label='Closing Price', color='black')
ax.plot(test_d.index, test_d['Open'], label='Opening Price', color='blue')
# Add a legend to the chart
ax.legend()
# Display grid lines on the chart
ax.grid()
# Adjust the layout for a better display
plt.tight_layout()
# Show the candlestick chart
plt.show()

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

#Define a function for multistep prediction using a trained model
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
        # and extract the value from the Numpy array
        predictions.append(scaler.inverse_transform(prediction)[0][0])

        # Append the prediction to the list of predictions
        predictions.append(prediction)

        # Update the current sequence by removing the first element and adding the prediction
        current_sequence = np.append(current_sequence[1:], prediction)

    return predictions

# Define a function for multistep prediction using linear regression
def multivariate_prediction(train_d, k):
    # Define the list of features to be used for the prediction
    features = ['Open', 'High', 'Low', 'Close', 'Adj Close',
                'Volume']

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

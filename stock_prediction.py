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
import pandas_datareader as web
import tensorflow as tf
import yfinance as yf
import os
import pickle

from datetime import datetime

from matplotlib.dates import DateFormatter, date2num
from mpl_finance import candlestick_ohlc
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, InputLayer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ------------------------------------------------------------------------------
# Load Data
## TO DO:
# 1) Check if data has been saved before. 
# If so, load the saved data
# If not, save the data into a directory
# ------------------------------------------------------------------------------

# Parameters

# TASK B.2 START LINE 38 - 118

DATA_SOURCE = 'yahoo'  # Directory where data will be saved or loaded from
TICK = 'AMZN'  # Stock ticker symbol
TRAIN_START = datetime.strptime('2020-01-01', '%Y-%m-%d').date()  # Start date for data retrieval
TRAIN_END = datetime.strptime('2023-08-08', '%Y-%m-%d').date()  # End date for data retrieval
VERSION = "3"  # Version of data to be saved
TRAIN_TEST_RATIO = 0.8  # Ratio of training data to total data
SPLIT_METHOD = 'date'  # Method for splitting data ('date' or 'random')
SCALE_FEATURES = False  # Whether to scale the feature columns ('False' or 'True')


# data = yf.download(TICK, start=TRAIN_START, end=TRAIN_END)

# Function to load or fetch data
def load_data(DATA_SOURCE, TICK, TRAIN_START, TRAIN_END, TRAIN_TEST_RATIO, SPLIT_METHOD, SCALE_FEATURES):
    # Check if the directory exists; if not, create it
    if not os.path.exists(DATA_SOURCE):
        os.makedirs(DATA_SOURCE)

    # Construct the file path for data saving/loading
    file_path = os.path.join(DATA_SOURCE, f"{TICK}_v{VERSION}.txt")

    # Check if saved data exists; if yes, load it; if not, fetch from Yahoo Finance
    if os.path.exists(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        print("DATA loaded successfully.")
    else:
        # Load data from Yahoo Finance
        data = yf.download(TICK, start=TRAIN_START, end=TRAIN_END)

        # Handle NaN values using forward filling
        data = data.ffill().dropna()

        # Save the fetched data
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        print("Data saved successfully.")

    # Split data into train and test sets based on the value {SPLIT_METHOD}
    if SPLIT_METHOD == 'date':
        # If split method is 'date', calculate the date for the end of the training time
        train_end_date = TRAIN_START + pd.DateOffset(days=int((TRAIN_END - TRAIN_START).days * TRAIN_TEST_RATIO))
        # Extract the data for the training period (from TRAIN_START to train_end_date)
        train_d = data[TRAIN_START:train_end_date]
        # Extract the data for the testing period (from train_end_date to TRAIN_END)
        test_d = data[train_end_date:TRAIN_END]
    elif SPLIT_METHOD == 'random':
        # If split method is 'random', use train_test_split to split the data randomly
        # The train_size parameter specifies the ratio of training data
        # The shuffle parameter is set too False to maintain order of the data

        train_d, test_d = train_test_split(data, train_size=TRAIN_TEST_RATIO, shuffle=False)
    else:
        # If an invalid split method is specified, raise a ValueError
        raise ValueError("Invalid split method specified.")

    # Scale feature columns if specified
    scalers = {}
    if SCALE_FEATURES:
        feature_columns = train_d.columns
        scaler = StandardScaler()  # Instantiate StandardScaler object, removing the mean, scaling to unit variance
        train_d[feature_columns] = scaler.fit_transform(train_d[feature_columns])  # Fit and transform training data
        test_d[feature_columns] = scaler.transform(test_d[feature_columns])  # Transform test data using the same scaler
        scalers['feature'] = scaler  # Store the scaler in the dictionary

    return train_d, test_d, scalers


# Load data using the load_data function
train_d, test_d, scalers = load_data(DATA_SOURCE, TICK, TRAIN_START, TRAIN_END, TRAIN_TEST_RATIO, SPLIT_METHOD,
                                     SCALE_FEATURES)
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
PRICE_VALUE = "Close"

scaler = MinMaxScaler(feature_range=(0, 1))
# Note that, by default, feature_range=(0, 1). Thus, if you want a different 
# feature_range (min,max) then you'll need to specify it here
scaled_data = scaler.fit_transform(train_d[PRICE_VALUE].values.reshape(-1, 1))
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

# Number of days to look back to base the prediction
PREDICTION_DAYS = 30  # Original

# To store the training data
x_train = []
y_train = []

scaled_data = scaled_data[:, 0]  # Turn the 2D array back to a 1D array
# Prepare the data
for x in range(PREDICTION_DAYS, len(scaled_data)):
    x_train.append(scaled_data[x - PREDICTION_DAYS:x])
    y_train.append(scaled_data[x])

# Convert them into an array
x_train, y_train = np.array(x_train), np.array(y_train)
# Now, x_train is a 2D array(p,q) where p = len(scaled_data) - PREDICTION_DAYS
# and q = PREDICTION_DAYS; while y_train is a 1D array(p)

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
# For som eadvances explanation of return_sequences:
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
TEST_START = '2022-01-02'
TEST_END = '2022-12-31'

test_data = yf.download(TICK, start=TRAIN_START, end=TRAIN_END, progress=False)

# The above bug is the reason for the following line of code
test_data = test_data[1:]

actual_prices = test_data[PRICE_VALUE].values

total_dataset = pd.concat((train_d[PRICE_VALUE], test_data[PRICE_VALUE]), axis=0)

model_inputs = total_dataset[len(total_dataset) - len(test_data) - PREDICTION_DAYS:].values
# We need to do the above because to predict the closing price of the fisrt
# PREDICTION_DAYS of the test period [TEST_START, TEST_END], we'll need the 
# data from the training period

model_inputs = model_inputs.reshape(-1, 1)
# TO DO: Explain the above line

model_inputs = scaler.transform(model_inputs)
# We again normalize our closing price data to fit them into the range (0,1)
# using the same scaler used above 
# However, there may be a problem: scaler was computed on the basis of
# the Max/Min of the stock price for the period [TRAIN_START, TRAIN_END],
# but there may be a lower/higher price during the test period 
# [TEST_START, TEST_END]. That can lead to out-of-bound values (negative and
# greater than one)
# We'll call this ISSUE #2

# TO DO: Generally, there is a better way to process the data so that we 
# can use part of it for training and the rest for testing. You need to 
# implement such a way

# ------------------------------------------------------------------------------
# Make predictions on test data
# ------------------------------------------------------------------------------
x_test = []
for x in range(PREDICTION_DAYS, len(model_inputs)):
    x_test.append(model_inputs[x - PREDICTION_DAYS:x, 0])

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
ax.set_title(f'Candlestick Chart for {TICK}')
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
            ######################
            # END OF CANDLESTICK #
            ######################

            ####################
            # START OF BOXPLOT #
            ####################
#Boxplot graph, using test data:
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
    ax.set_title(f'Boxplot Graph for {TICK}')
    # Ensure the chart layout is tight
    plt.tight_layout()
    # Show the boxplot chart
    plt.show()

# Call the function to plot the boxplot chart
plot_boxplot_chart(test_d)

            ##################
            # END OF BOXPLOT #
            ##################



#Original graph, with line tick plotting:
plt.plot(actual_prices, color="black", label=f"Actual {TICK} Price")
plt.plot(predicted_prices, color="red", label=f"Predicted {TICK} Price")
plt.title(f"{TICK} Share Price")
plt.xlabel("Time")
plt.ylabel(f"{TICK} Share Price")
plt.legend()
plt.show()

# ------------------------------------------------------------------------------
# Predict next day
# ------------------------------------------------------------------------------


real_data = [model_inputs[len(model_inputs) - PREDICTION_DAYS:, 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
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

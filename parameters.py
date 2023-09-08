# parameters.py
from datetime import datetime

# Data Source
data_source = 'yahoo'  # Directory where data will be saved or loaded from
tick = 'META'  # Stock ticker symbol
train_start = datetime.strptime('2015-01-01', '%Y-%m-%d').date()
train_end = datetime.strptime('2020-01-01', '%Y-%m-%d').date()
version = "1"  # Version of data to be saved
train_test_ratio = 0.8  # Ratio of training data to total data
split_method = 'random'  # Method for splitting data ('date' or 'random')
scale_features = False  # Whether to scale the feature columns ('False' or 'True')

# Prediction Parameters
price_value = "Close"
prediction_days = 20  # Number of days to look back to base the prediction

# Model Parameters
lstm_units = 50
dropout_rate = 0.2

# Training Parameters
num_epochs = 25
batch_size = 32

# Test Data Parameters
test_start = datetime.strptime('2020-01-02', '%Y-%m-%d').date()
test_end = datetime.strptime('2022-12-31', '%Y-%m-%d').date()

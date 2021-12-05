# Python script to set up a neural network with Keras for evaluating student test scores

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
# from fast_ml.model_development import train_valid_test_split
import numpy as np
import pandas as pd

# Encode categorical values in dataset
def encodeData(dataset):
    label_encoder = LabelEncoder()

    dataset['parental level of education'] = label_encoder.fit_transform(dataset['parental level of education'])
    dataset['lunch'] = label_encoder.fit_transform(dataset['lunch'])
    dataset['test preparation course'] = label_encoder.fit_transform(dataset['test preparation course'])
    dataset['race/ethnicity'] = label_encoder.fit_transform(dataset['race/ethnicity'])

    gender = pd.get_dummies(dataset['gender'], drop_first=True)
    dataset = pd.concat([dataset, gender], axis=1)
    dataset = dataset.drop('gender', axis=1)

    return dataset

# Read in student data from CSV
raw_dataset = pd.read_csv('src/StudentsPerformance.csv')
dataset = raw_dataset.copy()

# Encodes data from CSV
dataset = encodeData(dataset)

# Split training, test, and validation data.
# Splits the validation data using an extra train_test_split: https://datascience.stackexchange.com/questions/15135/train-test-validation-set-splitting-in-sklearn
x = dataset[['race/ethnicity', 'parental level of education', 'lunch', 'test preparation course', 'male']]
y = dataset[['math score', 'reading score', 'writing score']]
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.7, test_size=0.15, random_state=1)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.15, random_state=1)

# Scale the input values in the dataset
scaler = MinMaxScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Build the linear regression model
regression = LinearRegression()
regression.fit(x_train, y_train)
regression_prediction = regression.predict(x_test)

# Initializes a Sequential model with one input layer, three hidden layers, and one output layer.
# The layers use rectified linear activation as their activation function.
model = tf.keras.Sequential([
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(3, activation='relu')
])
model.compile(optimizer='adam', loss='mse')

# Stops training once val_loss has stopped decreasing
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=250)

# Trains the model
model.fit(
    x = x_train,
    y = y_train.values,
    epochs = 1000,
    validation_data = (x_test, y_test),
    verbose=1,
    batch_size = 64,
    callbacks = [early_stop]
)

# Report model analysis
print('Linear Regression Mean Absolute Error: ' + format(mean_absolute_error(y_test, regression_prediction)))
print('Linear Regression Mean Squared Error: ' + format(mean_squared_error(y_test, regression_prediction)))

# Report model accuracy on validation data
print('Evaluate model on validation data')
eval = model.evaluate(x_valid, y_valid, batch_size=128)
print(eval)

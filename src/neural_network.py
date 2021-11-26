# Python script to set up a neural network with Keras for evaluating student test scores

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import os

# One-hot encode categorical values from dataset
def encodeData(dataset):
    encode_gender = pd.get_dummies(dataset['gender'])
    dataset = dataset.join(encode_gender)
    dataset = dataset.drop('gender', axis=1)

    encode_ethnicity = pd.get_dummies(dataset['race/ethnicity'])
    dataset = dataset.join(encode_ethnicity)
    dataset = dataset.drop('race/ethnicity', axis=1)

    encode_education = pd.get_dummies(dataset['parental level of education'])
    dataset = dataset.join(encode_education)
    dataset = dataset.drop('parental level of education', axis=1)

    encode_lunch = pd.get_dummies(dataset['lunch'])
    dataset = dataset.join(encode_lunch)
    dataset = dataset.drop('lunch', axis=1)

    encode_test_prep = pd.get_dummies(dataset['test preparation course'])
    dataset = dataset.join(encode_test_prep)
    dataset = dataset.drop('test preparation course',axis=1)

    return dataset

# Read in student data from CSV
raw_dataset = pd.read_csv('src/StudentsPerformance.csv')
dataset = raw_dataset.copy()

# Encodes data from CSV
dataset = encodeData(dataset)

# Split and organize data into training and test sets
train_dataset = dataset.sample(frac=0.7, random_state=0)
test_dataset = dataset.drop(train_dataset.index)

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('math score')
test_labels = test_features.pop('math score')


model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="tanh"),
    tf.keras.layers.Dense(50, activation="tanh"),
    tf.keras.layers.Dense(2, activation="softmax")
])
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

# print(model.evaluate(train_dataset))


# print('Original Dataset')
# print(dataset.describe().transpose())
# print('Training Dataset')
# print(train_dataset.describe().transpose())
# print(dataset)

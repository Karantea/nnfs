import nnfs
import numpy as np
import math
from nnfs.datasets import vertical_data
from Chapter3 import Layer_Dense
from Chapter4 import Activation_Softmax
from Chapter4 import Activation_ReLU
from Chapter5 import Loss_CategoricalCrossEntropy

# Introduction to Optimization

# Method 1 (most simple one): change weights/biases random
# not reliable

nnfs.init()
X, y = vertical_data(samples=100, classes=3)

# create model
dense1 = Layer_Dense(2, 3)
activation1 = Activation_ReLU()
dense2 = Layer_Dense(3, 3)
activation2 = Activation_Softmax()

# create loss function
loss_function = Loss_CategoricalCrossEntropy()

# variables to track loss
# copy() ensures full copy rather than reference to object
lowest_loss = 9999999 # init
best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

# iteration for desired time

for iteration in range(1000):
    # generate new set of weights/biases
    dense1.weights = 0.05 * np.random.randn(2, 3)
    dense1.biases = 0.05 * np.random.randn(1, 3)
    dense2.weights = 0.05 * np.random.randn(3, 3)
    dense2.biases = 0.05 * np.random.randn(1, 3)

    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accurary = np.mean(predictions == y)

    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accurary)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

# Method 2: Update weights with small adjustments
# not viable for high complexity problems
# problem: local minimum of loss

for iteration in range(1000):
    # generate new set of weights/biases
    dense1.weights += 0.05 * np.random.randn(2, 3)
    dense1.biases += 0.05 * np.random.randn(1, 3)
    dense2.weights += 0.05 * np.random.randn(3, 3)
    dense2.biases += 0.05 * np.random.randn(1, 3)

    # forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)
    loss = loss_function.calculate(activation2.output, y)

    # calculate accuracy
    predictions = np.argmax(activation2.output, axis=1)
    accurary = np.mean(predictions==y)

    if loss < lowest_loss:
        print('New set of weights found, iteration:', iteration,
              'loss:', loss, 'acc:', accurary)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1.biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss
    else:
        # revert weights and biases
        dense1.weights = best_dense1_weights.copy()
        dense1.biases = best_dense1_biases.copy()
        dense2.weights = best_dense2_weights.copy()
        dense2.biases = best_dense2_biases.copy()

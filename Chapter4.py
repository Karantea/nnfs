import numpy as np
import math
from nnfs.datasets import spiral_data
from Chapter3 import Layer_Dense

class Activation_ReLU:

    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        self.inputs = inputs

    def backward(self, dvalues):
        # needs modification, so we copy values
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0

class Activation_Softmax:

    def forward(self, inputs):

        # Get unnormalized probabilities
        exp_values = np.exp(inputs - np.max(inputs, axis=1,
                                            keepdims=True))
        # Normalize them for each sample
        probabilities = exp_values / np.sum(exp_values, axis=1,
                                            keepdims=True)
        self.output = probabilities

    def backwards(self, dvalues):

        # create uninizialized array
        self.dinputs = np.empty_like(dvalues)

        # enumerate outputs and gradients
        for index, (single_output, single_davalue) in \
            enumerate(zip(self.output, dvalues)):
            # flatten output array
            single_output = single_output.reshape(-1, 1)
            # calculate jacobian matrix of the output
            jacobian_matrix = np.diagflat(single_output) - \
                              np.dot(single_output, single_output.T)
            # calculate sample-wise gradient
            # and add it to the array of sample gradients
            self.dinputs[index] = np.dot(jacobian_matrix, single_davalue)


if __name__ == "__main__":
    # Activation Functions
    # Rectified linear function -> unbounded, not normalized with other units
    # exclusive
    inputs = [0, 2, -2, 3.3, -2,7, 1.1, 2.2, -100]
    output = np.maximum(0, inputs)

    # example
    X, y = spiral_data(samples=100, classes=3)
    dense1 = Layer_Dense(2, 3)
    activation1 = Activation_ReLU()
    # forward pass through layer
    dense1.forward(X)
    # pass through activation function with output from previous layer
    activation1.forward(dense1.output)

    # Softmax Acitvation function
    # taked non-normalized, uncalibrated inputs
    # produces normalized probabilities for classes
    # exponentation to get positive values only and stability
    # normalization with the sum of exponentiated values
    # axis=0 (row-wise), axis=1 (column-wise)

    # example
    X, y = spiral_data(samples=100, classes=3)
    # dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2, 3)
    # ReLU activation with dense layer
    activation1 = Activation_ReLU()
    # second dense layer with 3 input features (output from dense1)
    # and 3 outputs values
    dense2 = Layer_Dense(3, 3)
    # softmax activation
    activation2 = Activation_Softmax()

    # forward pass through dense layer 1, ReLU, dense layer 2 and softmax
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    print(activation2.output[:5])
    # [[0.33333333 0.33333333 0.33333333]
    #  [0.33333295 0.33333364 0.33333341]
    #  [0.33333221 0.33333385 0.33333394]
    #  [0.33333169 0.33333409 0.33333422]
    #  [0.33333127 0.33333428 0.33333444]]
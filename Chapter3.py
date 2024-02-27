import numpy as np
import nnfs
from nnfs.datasets import spiral_data
import matplotlib.pyplot as plt

# Dense Layer Class (fully connected)
class Layer_Dense:

    def __init__(self, n_inputs, n_neurons,
                 weight_regularizer_l1=0, weight_regularizer_l2=0,
                 bias_regularizer_l1=0, bias_regularizer_l2=0):
        # Initialize weights and biases (oftentimes randomly, pretrained possible)
        # inputs x neurons instead of transposing in every forward pass
        self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)  # gaussian distribution (m=0, v=1)
        self.biases = np.zeros((1, n_neurons))
        # Set regularization strength (lambda)
        self.weight_regulizer_l1 = weight_regularizer_l1
        self.weight_regulizer_l2 = weight_regularizer_l2
        self.bias_regulizer_l1 = bias_regularizer_l1
        self.bias_regulizer_l2 = bias_regularizer_l2


    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.inputs = inputs
        self.output = np.dot(inputs, self.weights) + self.biases

    def backward(self, dvalues):
        # Gradients on parameters
        self.dweights = np.dot(self.inputs.T, dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        # Gradients on regularization
        # L1 on weights
        if self.weight_regulizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.dweights += self.weight_regulizer_l1 * dL1
        # L2 on weights
        if self.weight_regulizer_l2 > 0:
            self.dweights += self.weight_regulizer_l2 * self.weights
        # L1 on biases
        if self.bias_regulizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.dbiases += self.bias_regulizer_l1 * dL1
        # L2 on biases
        if self.bias_regulizer_l2 > 0:
            self.dbiases += self.bias_regulizer_l2 * self.biases
        # Gradients on values
        self.dinputs = np.dot(dvalues, self.weights.T)



if __name__ == "__main__":
    # Adding Layers
    inputs = [[1, 2, 3, 2.5],
              [2., 5., - 1., 2],
              [- 1.5, 2.7, 3.3, - 0.8]]
    #Neuron 1
    weights = [[0.2, 0.8, - 0.5, 1],
               [0.5, - 0.91, 0.26, - 0.5],
               [- 0.26, - 0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    # Neuron 2
    weights2 = [[0.1, - 0.14, 0.5],
                [- 0.5, 0.12, - 0.33],
                [- 0.44, 0.73, - 0.13]]
    biases2 = [- 1, 2, - 0.5]

    layer1_output = np.dot(inputs, np.array(weights).T) + biases
    layer2_output = np.dot(layer1_output, np.array(weights2).T) + biases2

    # for reproducability purposes
    nnfs.init()
    X, y = spiral_data(samples=100, classes=3)
    plt.scatter(X[:,0], X[:,1], c=y, cmap='brg')
    plt.show()

    # example
    X, y = spiral_data(samples=100, classes=3)
    # create dense layer with 2 input features and 3 output values
    dense1 = Layer_Dense(2,3)
    # forward pass
    dense1.forwards(X)
    print(dense1.weights)
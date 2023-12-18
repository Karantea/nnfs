import numpy as np
import nnfs
from nnfs.datasets import spiral_data
from Chapter3 import Layer_Dense
from Chapter4 import Activation_ReLU
from Chapter9 import Activation_Softmax_Loss_CategoricalCrossentropy


# Optimizers
# Stochastic Gradient Descent: merges all types of optimizers
# regardless of input data (single/batch/whole)

class Optimizer_SGD:
    # initialization
    # learning rate 1.0 as default
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / (1. + self.decay * self.iterations))

    # updating parameters
    def update_params(self, layer):
        if self.momentum:
            # if layer does not contain momentum arrays
            # create them filled with zeroes
            if not hasattr(layer, 'weight_momentum'):
                # no array for weights = no array for biases
                layer.weight_momentum = np.zeros_like(layer.weights)
                layer.bias_momentum = np.zeros_like(layer.biases)

            # weight updates with momentum
            # previous updates multiplied by retain factor and update
            # with current gradients
            weight_updates = \
                self.momentum * layer.weight_momentum - \
                self.current_learning_rate * layer.dweights
            layer.weight_momentum = weight_updates

            # build bias updates
            bias_updates = \
                self.momentum * layer.bias_momentum - \
                self.current_learning_rate * layer.dbiases
            layer.bias_momentum = bias_updates

        else:
            # vanilla SGD updates
            weight_updates = -self.learning_rate * layer.dweights
            bias_updates = -self.learning_rate * layer.dbiases

        # update weights
        layer.weights += weight_updates
        layer.biases += bias_updates

    # call once after any parameter updates
    def post_update_params(self):
        self.iterations += 1


# example: 1x64 neural network

X, y = spiral_data(samples=100, classes=3)

# dense layer with 2 input features and 64 output values
dense1 = Layer_Dense(2, 64)
# ReLU activation
activation1 = Activation_ReLU()
# second layer with 64 input features and 3 output values
dense2 = Layer_Dense(64, 3)
# softmax classifier combined loss and activation
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
# create optimizer
# decay 1e-2 to fast
optimizer = Optimizer_SGD(decay=1e-3, momentum=0.)

for epoch in range(10000):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    # calculate accuracy from output and targets
    predictions = np.argmax(loss_activation.output, axis=1)
    # if len(y.shape) == 2:
    #    y = np.argmax(y, axis=1)
    accuracy = np.mean(predictions == y)

    if not epoch % 100:
        print(f'epoch: {epoch}, ' +
              f'acc: {accuracy:.3f}, ' +
              f'loss: {loss:.3f}',
              f'lr: {optimizer.current_learning_rate}')

    # Backpropagation
    loss_activation.backward(loss_activation.output, y)
    dense2.backward(loss_activation.dinputs)
    activation1.backward(dense2.dinputs)
    dense1.backward(activation1.dinputs)

    # update weights and biases
    optimizer.pre_update_params()
    optimizer.update_params(dense1)
    optimizer.update_params(dense2)
    optimizer.post_update_params()

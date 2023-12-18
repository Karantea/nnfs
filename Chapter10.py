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
    def __init__(self, learning_rate=1., decay=0., momentum=0.):
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


# Part 2: Adaptive Gradient

class Optimizer_AdaGrad:

    # initialization
    def __init__(self, learning_rate=1., decay=0., epsilon=1e-7):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / 1. + self.decay * self.iterations)

    # update parameters
    def update_params(self, layer):
        # if layer does not have cache array, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update weights with sqared current gradients
        layer.weight_cache += layer.dweights ** 2
        layer.bias_cache += layer.dbiases ** 2

        # vanilla SGD + normalization
        # with sqare rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# Part 3: RMSProp

class Optimizer_RMSprop:

    # initialization
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7,
                 rho=.9):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.rho = rho

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / 1. + self.decay * self.iterations)

    # update parameters
    def update_params(self, layer):
        # if layer does not have cache array, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update weights with squared current gradients
        layer.weight_cache = self.rho * layer.weight_cache + \
                              (1 - self.rho) * layer.dweights ** 2
        layer.bias_cache = self.rho * layer.bias_cache + \
                           (1 - self.rho) * layer.dbiases**2

        # vanilla SGD + normalization
        # with sqare rooted cache
        layer.weights += -self.current_learning_rate * \
                         layer.dweights / \
                         (np.sqrt(layer.weight_cache) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        layer.dbiases / \
                        (np.sqrt(layer.bias_cache) + self.epsilon)

    def post_update_params(self):
        self.iterations += 1


# Part 4: Adam

class Optimizer_Adam:

    # initialization
    def __init__(self, learning_rate=.001, decay=0., epsilon=1e-7,
                 beta_1=.9, beta_2=.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    # call once before any parameter updates
    def pre_update_params(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * \
                                         (1. / 1. + self.decay * self.iterations)

    # update parameters
    def update_params(self, layer):
        # if layer does not have cache array, create them
        if not hasattr(layer, 'weight_cache'):
            layer.weight_momentum = np.zeros_like(layer.weights)
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.bias_momentum = np.zeros_like(layer.biases)
            layer.bias_cache = np.zeros_like(layer.biases)

        # update momentum with current gradients
        layer.weight_momentum = self.beta_1 * \
                                layer.weight_momentum + \
                                (1 - self.beta_1) * layer.dweights

        layer.bias_momentum = self.beta_1 * \
                              layer.weight_momentum + \
                              (1 - self.beta_1) * layer.dbiases

        # get corrected momentums
        # self.iterations is 0 at first
        # but 1 is needed
        weight_momentum_corrected = layer.weight_momentum / \
                                    (1 - self.beta_1 ** (self.iterations + 1))
        bias_momentum_corrected = layer.bias_momentum / \
                                  (1 - self.beta_1 ** (self.iterations + 1))

        # update cache with squared current gradient
        # beta_2 = rho in RMSprop
        layer.weight_cache = self.beta_2 * layer.weight_cache + \
                             (1 - self.beta_2) * layer.dweights**2
        layer.bias_cache = self.beta_2 * layer.bias_cache + \
                           (1 - self.beta_2) * layer.dbiases**2

        # get corrected cache
        weight_cache_corrected = layer.weight_cache / \
                                 (1 - self.beta_2 ** (self.iterations + 1))
        bias_cache_corrected = layer.bias_cache / \
                               (1 - self.beta_2 ** (self.iterations + 1))

        # vanilla SGD + normalization
        # with square rooted cache
        layer.weights += -self.current_learning_rate * \
                         weight_momentum_corrected / \
                         (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * \
                        bias_momentum_corrected / \
                        (np.sqrt(bias_cache_corrected) + self.epsilon)


    def post_update_params(self):
        self.iterations += 1


# example: 1x64 neural network with Stochastic Gradient Descent

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

# Example with Adaptive Gradient or
# Root Mean Square Propagation

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
# Choice: AdaGrad/RMSprop/Adam
#optimizer = Optimizer_AdaGrad(decay=1e-4)
#optimizer = Optimizer_RMSprop(learning_rate=0.02,
#                              decay=1e-5,
#                              rho=0.999)
optimizer = Optimizer_Adam(learning_rate=0.05,
                           decay=1e-7)
for epoch in range(10000):
    # Forward pass
    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    loss = loss_activation.forward(dense2.output, y)
    # calculate accuracy from output and targets
    predictions = np.argmax(loss_activation.output, axis=1)
    if len(y.shape) == 2:
        y = np.argmax(y, axis=1)
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

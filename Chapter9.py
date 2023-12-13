import nnfs
import numpy as np
import math
from nnfs.datasets import vertical_data
from Chapter3 import Layer_Dense
from Chapter4 import Activation_Softmax
from Chapter4 import Activation_ReLU
from Chapter5 import Loss_CategoricalCrossEntropy

# Backpropagation
# simple example for 1 neuron only

x = [1.0, -2.0, 3.0]
w = [-3.0, -1.0, 2.0]
b = 1.0

# -3.0
xw0 = x[0] * w[0]
# 2.0
xw1 = x[1] * w[1]
# 6.0
xw2 = x[2] * w[2]

# adding bias
# 6.0
z = xw0 + xw1 + xw2 + b

# adding ReLU activation function
# 6.0
y = max(z, 0)

# Backward pass
# derivative from next layer
dvalue = 1.0

# Derivative of ReLU and chain rule
# 1.0
drelu_dz = dvalue * (1. if z > 0 else 0.)

# function before ReLU is the sum of weighted inputs and bias
# calculation of partial derivaties with respect to each weighted input + bias
# multiplied by partial derivative of subsequent function (ReLU)

dsum_dxw0 = 1
# 1.0 * 1.0
drelu_dxw0 = drelu_dz * dsum_dxw0
dsum_dxw1 = 1
drelu_dxw1 = drelu_dz * dsum_dxw1
dsum_dxw2 = 1
drelu_dxw2 = drelu_dz * dsum_dxw2
dsum_db = 1
drelu_db = drelu_dz * dsum_db

# Backward pass
# derivative of next function: multiplication

dmul_dx0 = w[0]
# -3.0
drelu_dx0 = drelu_dxw0 * dmul_dx0
dmul_dw0 = x[0]
# 1.0
drelu_dw0 = drelu_dxw0 * dmul_dw0
dmul_dx1 = w[1]
# -1.0
drelu_dx1 = drelu_dxw1 * dmul_dx1
dmul_dw1 = x[1]
# -2.0
drelu_dw1 = drelu_dxw1 * dmul_dw1
dmul_dx2 = w[2]
# 2.0
drelu_dx2 = drelu_dxw2 * dmul_dx2
dmul_dw2 = x[2]
# 3.0
drelu_dw2 = drelu_dxw2 * dmul_dw2

# simplifying code

drelu_dx0 = dvalue * (1. if z > 0 else 0.) * w[0]
drelu_dw0 = dvalue * (1. if z > 0 else 0.) * x[0]

dx = [drelu_dx0, drelu_dx1, drelu_dx2]
dw = [drelu_dw0, drelu_dw1, drelu_dw2]
db = drelu_db

# optimizer: applying gradients to weights to minimize output
# changing weight values:

# [-3.001, -0.998, 1.997], 0.999
w[0] += -0.001 * dw[0]
w[1] += -0.001 * dw[1]
w[2] += -0.001 * dw[2]
b += -0.001 * b

# adding: 5.985
# exercise: decreasing neuron output is not sensible (loss needs to be decreased)
z = xw0 + xw1 + xw2 + b


### Using a neuron layer insteadt of a single neuron as above

# passed in gradient from next layer
# derivative with respect to inputs

dvalues = np.array([[1., 1., 1.],
                    [2., 2., 2.0],
                    [3., 3., 3.]])

# 3 sets of weights, one for each neuron
# 4 inputs = 4 weights
# keep them transposed

weights = np.array([[0.2, 0.8, -0.5, 1.],
                    [0.5, -0.91, 0.26, -0.5],
                    [-0.26, -0.27, 0.17, 0.87]]).T

# gradient of neuron function with respect to inputs
#dx0 = sum(weights[0] * dvalues[0])
#dx1 = sum(weights[1] * dvalues[0])
#dx2 = sum(weights[2] * dvalues[0])
#dx3 = sum(weights[3] * dvalues[0])

#dinputs = np.array([dx0, dx1, dx2, dx3])
# simplify
# [[0.44, -0.38, -0.07, 1.37], [0.88, -0.76. -0.14, 2.74], [1.32, -1.14, -0.21, 4.11]]
dinputs = np.dot(dvalues, weights.T)

# derivative with respect to weights

inputs = np.array([[1, 2, 3, 2.5],
                   [2., 5., -1., 2],
                   [-1.5, 2.7, 3.3, -0.8]])
# [[0.5, 0.5, 0.5], [20.1, 20.1, 20.1], [10.9, 10.9, 10.9], [4.1, 4.1, 4.1]]
dweights = np.dot(inputs.T, dvalues)

# derivative with respect to biases, always 1 multiplied by incoming gradients
# sum with neurons column-wise

biases = np.array([[2, 3, 0.5]])
# [[6., 6., 6.]]
dbiases = np.sum(dvalues, axis=0, keepdims=True)

# derivative of ReLU function (1 if z > 0)
# example layer output

z = np.array([[1, 2, -3, -4],
              [2, -7, -1, 3],
              [-1, 2, 5, -1]])
# example gradients
# in backpropagation, ReLU receives gradient of same shape as layer output

dvalues_z = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12]])
# relu derivative
# drelu =  np.zeros_like(z)
# drelu[z > 0] = 1
# chain rule
# [[1, 2, 0, 0], [5, 0, 0, 8], [0, 10, 11, 0]]
# drelu *= dvalues_z
# simplification

drelu = np.copy(dvalues_z)
drelu[z <= 0] = 0
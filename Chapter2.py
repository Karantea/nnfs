import numpy as np
import matplotlib as plt


if __name__ == "__main__":
    # Chapter 2: Coding our first Neurons
    # 1 neuron (= 1 bias), 4 inputs
    # weights (trainable parameters to adjust) get assigned
    # randomly most times
    inputs = [1,2,3, 2.5]
    weights = [0.2, 0.8, -0.5, 1]
    bias = 2

    # weighted sum [4.8]
    output = (inputs[0]*weights[0]+
              inputs[1]*weights[1]+
              inputs[2]*weights[2] +
              inputs[3]*weights[3] + bias)

    # A Layer of Neurons
    # 3 neurons, 4 inputs, [4.8, 1.21, 2.385]
    weights1 = [0.2, 0.8, -0.5, 1]
    weights2 = [0.5, -0.91, 0.26, -0.5]
    weights3 = [-0.26, -0.27, 0.17, 0.87]

    bias1 = 2
    bias2 = 3
    bias3 = 0.5

    outputs = [
        #neuron 1
        inputs[0]*weights1[0] +
        inputs[1]*weights1[1] +
        inputs[2]*weights1[2] +
        inputs[3]*weights1[3] + bias1,

        #neuron 2
        inputs[0]*weights2[0] +
        inputs[1]*weights2[1] +
        inputs[2]*weights2[2] +
        inputs[3]*weights2[3] + bias2,

        #neuron 3
        inputs[0]*weights3[0] +
        inputs[1]*weights3[1] +
        inputs[2]*weights3[2] +
        inputs[3]*weights3[3] + bias3
    ]

    #automating code
    weightsAll = [[0.2, 0.8, -0.5, 1],
                  [0.5, -0.91, 0.26, -0.5],
                  [-0.26, -0.27, 0.17, 0.87]]
    biases = [2, 3, 0.5]
    layers_output = []

    for neuron_weights, neuron_bias in zip(weightsAll, biases):
        neuron_output = 0
        for n_input, weight in zip(inputs, neuron_weights):
            # multiplying input with weight
            neuron_output += n_input*weight
        # adding bias
        neuron_output += neuron_bias
        layers_output.append(neuron_output)

    # Tensors, Arrays, Vectors
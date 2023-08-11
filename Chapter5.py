import numpy as np
import math
from nnfs.datasets import spiral_data
from Chapter3 import Layer_Dense
from Chapter4 import Activation_Softmax
from Chapter4 import Activation_ReLU

# Calculating Network Error with Loss
# Categorical Cross Entropy Loss

# example output
softmax_output = [0.7, 0.1, 0.2]
# ground truth
target_output = [1, 0, 0]

# loss = 0.35667494393873245
loss = -(math.log(softmax_output[0])*target_output[0] +
         math.log(softmax_output[1])*target_output[1] +
         math.log(softmax_output[2])*target_output[2])

# with assumptions regarding one-hot target vector
# loss2 = 0.35667494393873245
loss2 = -math.log(softmax_output[0])


# Update: batches of softmax output distributions + dynamic to target index

softmax_outputs = np.array([[0.7, 0.1, 0.2],
                            [0.1, 0.5, 0.4],
                            [0.02, 0.9, 0.08]])
# ground truth (from classes=["dog", "cat", "human"])
class_target = [0, 1, 1]
# losses = [0.35667494 0.69314718 0.10536052]
losses = -np.log(softmax_outputs[range(len(softmax_outputs)), class_target])
# average_loss = 0.38506088005216804
average_loss = np.mean(losses)

# adding check for one-hot array or sparse (like above)

class_targets = np.array([[1, 0, 0],
                          [0, 1, 0],
                          [0, 1, 1]])
# probabilities for target values (if categorical)
if len(class_targets.shape) == 1: # target sparse
    correct_confidences = softmax_outputs[
        range(len(softmax_outputs)),
        class_targets
    ]
elif len(class_targets.shape) == 2: # target one-hot vectors
    correct_confidences = np.sum(
        softmax_outputs * class_targets,
        axis=1
    )
# clipping to prevent log(0)
correct_confidences = np.clip(correct_confidences, 1e-7, 1-1e-7)
print(correct_confidences)
neg_log = -np.log(correct_confidences)
average_losses = np.mean(neg_log)
print(average_losses)


# The Categorical Cross-Entropy Loss Class (common)

class Loss:
    # Calculates the data and regularization losses given model
    # output and ground truth values
    def calculate(self, output, y):
        # calculate sample losses
        samples_losses = self.forward(output, y)

        # calculate mean loss
        data_loss = np.mean(samples_losses)

        # return loss
        return data_loss

# Cross-entropy loss
class Loss_CategoricalCrossEntropy(Loss):

    # forward pass
    def forward(self, y_pred, y_true):
        # number of samples in batch
        samples = len(y_pred)
        # clip data to prevent division by 0
        # clip both sides to not drag down mean to any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        # propabilities for target values
        # 1) categorical labels
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true]
        # 2) one-hot encoded labels
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis=1
            )

        # Losses
        negative_log_likelihoods = -np.log(correct_confidences)
        return negative_log_likelihoods


# example
# create dataset
X, y = spiral_data(samples=100, classes=3)
# create dense layer with 2 input features and 3 output values
dense1 = Layer_Dense(2, 3)
# create ReLU activation (to be used with dense layer)
activation1 = Activation_ReLU()
# create second dense layer with 3 input features and 3 output values
dense2 = Layer_Dense(3, 3)
# create softmax activation
activation2 = Activation_Softmax()
# create loss function
loss_function = Loss_CategoricalCrossEntropy()

# perform forward pass
dense1.forward(X)
# forward pass through activation function
activation1.forward(dense1.output)
# forward pass through dense layer
dense2.forward(activation1.output)
activation2.forward(dense2.output)
loss = loss_function.calculate(activation2.output, y)
# calculate accuracy from output along first axix
predictions = np.argmax(activation2.output, axis=1)
if len(y.shape) == 2:
    # converting one-hot encoded targets to sparse values
    y = np.argmax(y, axis=1)
accuracy = np.mean(predictions == y)
print('acc:', accuracy)

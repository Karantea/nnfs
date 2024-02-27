

# Chapter 14: L1 and L2 Regularization
# Penalty calculation to penalize for too large weights and biases
# to avoid overfitting
# L1: linear, L2: nonlinear (sum of squared weights and biases)

# adding regulatization to training loop: blueprint

data_loss = loss_function.forward(activation2.output, y)
regularization_loss = loss_function.regularization_loss(dense1) +
                      loss_function.regularization_loss(dense2)

loss = data_loss + regularization_loss



# Chapter 15: Dropout
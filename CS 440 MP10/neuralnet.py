# neuralnet.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Justin Lizama (jlizama2@illinois.edu) on 10/29/2019
# Modified by James Soole for the Fall 2023 semester

"""
This is the main entry point for MP10. You should only modify code within this file.
The unrevised staff files will be used for all other files and classes when code is run, 
so be careful to not modify anything else.
"""

# Here we are importing all the required libraries for this assignment.
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import get_dataset_from_arrays
from torch.utils.data import DataLoader

# Here we are creating a class called NeuralNet.
class NeuralNet(nn.Module):

    # Here we are creating an instance of the class called NeuralNet.
    def __init__(self, lrate, loss_fn, in_size, out_size):
        """
        Initializes the layers of your neural network.

        @param lrate: learning rate for the model
        @param loss_fn: A loss function defined as follows:
            @param yhat - an (N,out_size) Tensor
            @param y - an (N,) Tensor
            @return l(x,y) an () Tensor that is the mean loss
        @param in_size: input dimension
        @param out_size: output dimension

        For Part 1 the network should have the following architecture (in terms of hidden units):
        in_size -> h -> out_size , where  1 <= h <= 256
        
        We recommend setting lrate to 0.01 for part 1.

        """

        # Here we are initializing the parent class (usually torch.nn.Module) for the NeuralNet class.
        super(NeuralNet, self).__init__()

        # Here we are assigning the provided loss function to the 'loss_fn' attribute of the instance.
        self.loss_fn = loss_fn

        # Here we are creating the nn.Sequential and setting it equal to self.layers.
        self.layers = nn.Sequential(

            # Here is the 2D convolution layer with 3 input channels, 16 output channels, and a 3x3 kernel without padding.
            nn.Conv2d(3, 16, kernel_size = (3, 3), padding = 0),

            nn.ReLU(),   # Here is the rectified Linear Activation function.

            nn.MaxPool2d((2, 2)),   # Here is the 2x2 max pooling layer for spatial down-sampling.

            # Here is the 2D convolution layer with 16 input channels, 32 output channels, and a 3x3 kernel without padding.
            nn.Conv2d(16, 32, kernel_size = (3, 3), padding = 0),

            nn.ReLU(),   # Here is the rectified Linear Activation function.

            nn.ReLU(),   # Here is the rectified Linear Activation function.

            nn.Flatten(),   # Here is the faltten layer flattens the tensor into a 1D tensor.

            # Here is the fully connected linear layer with input features of size 32*12*12 and 64 output features.
            nn.Linear(32 * 12 * 12, 64),

            nn.ReLU(),   # Here is the rectified Linear Activation function.

            nn.Linear(64, 4)   # Here is the fully connected linear layer with 64 input features and 4 output features.
        )

        # Here we are initializing Stochastic Gradient Descent optimizer with specified learning rate and momentum.
        self.optim = optim.SGD(self.parameters(), lr = lrate, momentum = 0.9)

    # Here we are defining a function called forward.
    def forward(self, x):
        """
        Performs a forward pass through your neural net (evaluates f(x)).

        @param x: an (N, in_size) Tensor
        @return y: an (N, out_size) Tensor of output from the network
        """

        # Here we are returning the input tensor that is reshaped to (x.size(0), 3, 31, 31)).
        return self.layers((x.view(x.size(0), 3, 31, 31)))

    # Here we are defining a function called step.
    def step(self, x, y):
        """
        Performs one gradient step through a batch of data x with labels y.

        @param x: an (N, in_size) Tensor
        @param y: an (N,) Tensor
        @return L: total empirical risk (mean of losses) for this batch as a float (scalar)
        """

        self.optim.zero_grad()   # Here we are zeroing out any gradients from previous iterations.

        # Here we are computing the loss by comparing the model's prediction to the true value.
        result = self.loss_fn(self.forward(x), y)

        result.backward()   # Here we are computing the gradients for each model parameter with respect to the loss.

        self.optim.step()   # Here we are updating the model's parameters using the computed gradients.

        return result.item()   # Here we are returning the computed loss as a scalar value.

# Here we are defining a function called normalize.
def normalize(calculator, calculateOne = None, calculateTwo = None):

    # Here we are checking if 'calculateOne' is not provided, compute the mean of 'calculator' across the 0th dimension, else use the provided 'calculateOne'.
    calculateOne = calculator.mean(dim = 0, keepdim = True) if calculateOne is None else calculateOne

    # Here we are checking if 'calculateTwo' is not provided, compute the mean of 'calculator' across the 0th dimension, else use the provided 'calculateTwo'.
    calculateTwo = calculator.std(dim = 0, keepdim = True) if calculateTwo is None else calculateTwo

    # Here we are returning the normalized calculator by subtracting 'calculateOne' and dividing by 'calculateTwo'
    return (calculator - calculateOne) / calculateTwo

# Here we are defining a function called fit.
def fit(train_set,train_labels,dev_set,epochs,batch_size=100):
    """ 
    Make NeuralNet object 'net'. Use net.step() to train a neural net
    and net(x) to evaluate the neural net.

    @param train_set: an (N, in_size) Tensor
    @param train_labels: an (N,) Tensor
    @param dev_set: an (M,) Tensor
    @param epochs: an int, the number of epochs of training
    @param batch_size: size of each batch to train on. (default 100)

    This method *must* work for arbitrary M and N.

    The model's performance could be sensitive to the choice of learning rate.
    We recommend trying different values in case your first choice does not seem to work well.

    @return losses: list of floats containing the total loss at the beginning and after each epoch.
        Ensure that len(losses) == epochs.
    @return yhats: an (M,) NumPy array of binary labels for dev_set
    @return net: a NeuralNet object
    """

    # Here we are normalizing the training dataset.
    normalizedTrainSet = normalize(train_set)

    losses = []   # Here we are initializing an empty list to store losses.

    # Here we are instantiating a neural network with specified learning rate, loss function, input size, and number of layers.
    net = NeuralNet(0.003, nn.CrossEntropyLoss(), train_set.size(dim = 1), 4)

    # Here we are training the network for a specified number of epochs.
    for each in range(epochs):

        # Here we are calculating the total loss for the current epoch by iterating through the dataset batches.
        totalLoss = sum(net.step(eachOne['features'], eachOne['labels']) for eachOne in DataLoader(get_dataset_from_arrays(normalizedTrainSet, train_labels), batch_size, shuffle = True))

        losses.append(totalLoss)   # Here we are appending the total loss of the current epoch to the losses list.

    # Here we are predicting the labels of the development set using the trained neural network.
    yhats = net(normalize(dev_set, calculateOne = normalizedTrainSet.mean(), calculateTwo = normalizedTrainSet.std())).argmax(1)

    # Here is the yhats variable that we will return.
    yhats = yhats.detach().cpu().numpy().astype(int)

    # Here we are returning the losses list, the predicted labels converted to CPU integers, and the trained neural network.
    return losses, yhats, net

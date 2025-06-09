

import torch 
import torch.nn as nn
import torch.nn.functional as F


class Log_reg_one_neuron(nn.Module):
    """
    Simple possible neural network : just one neuron
    We will be used to predict the score of one candidates 

    Output : sigmoid(Ax+b) with x the features of each town 
    """
    def __init__(self, input_dim):
        super(Log_reg_one_neuron, self).__init__()
        self.out = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.out(x)        
        x = nn.Sigmoid()(x)
        return x


class NN_2Layers_1Neuron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(NN_2Layers_1Neuron, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        x = nn.Sigmoid()(x)
        return x


class NN2Layers_no_softmax(nn.Module):
    """
    Neural network with just one hidden layer 
    We will be used to predict simultaneous the score of the 11 candidates

    To output a probability distribution :
        we divided each neuron output by the sum of all neuron output 
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN2Layers_no_softmax, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)
    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        x = torch.abs(x)
        s = torch.sum(x, dim=1).view(-1,1)
        x = x / s
        
        return x


class NN2Layers_with_softmax(nn.Module):
    """
    Neural network with just one hidden layer 
    We will be used to predict simultaneous the score of the 11 candidates

    To output a probability distribution :
        this time, we apply softmax on the output layer 
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(NN2Layers_with_softmax, self).__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.hidden(x))
        x = self.out(x)
        
        x = nn.Softmax(dim=1)(x)
        return x


class NN1Layer(nn.Module):
    """
    Neural network with just one output layer (11 neurons)
    We will be used to embed the 11 candidates
    """
    def __init__(self, input_dim, output_dim):
        super(NN1Layer, self).__init__()
        self.out = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        x = self.out(x)
        x = nn.Softmax(dim=1)(x)
        
        return x


            
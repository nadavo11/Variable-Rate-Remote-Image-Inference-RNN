import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import *


class BinaryLayer(Function):
    def forward(self, x):
        return torch.sign(x)

    def backward(self, grad_output):
        return grad_output


class BinaryLayer2(nn.Module):


    """ --------
    this guy  didn't implement a working binary layer, nor did he specify how
     it should work, so I have to do it myself.
      -------"""

    """Binary layer as defined in the paper: STILL NOT WORKING!!!1 :("""
    # TODO: Make this work
    def forward(self, x):
        # noise
        probs_tensor = torch.rand(x.size())
        #
        errors = Variable(torch.FloatTensor(x.size()))
        # shift x to be in [0:1]
        probs_threshold = torch.div(torch.add(x, 1), 2)
        # calculate the indices effected by the noise
        alpha = 1-x[probs_tensor <= probs_threshold.data]
        beta = -x[probs_tensor > probs_threshold.data] - 1
        # get the noise
        errors[probs_tensor <= probs_threshold.data] = alpha
        errors[probs_tensor > probs_threshold.data] = beta

        y = x + errors
        return y

    def backward(self, grad_output):
        return grad_output
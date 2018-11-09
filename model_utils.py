import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.utils import data as tud
from torchvision import models as vmodels
import warpctc_pytorch
import ctcdecode
import torch.nn.utils.rnn as rnn_utils

import numpy as np
import math


from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def init_weights(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.normal_(module.bias.data)
        print('initialized Linear')

    elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight, mode='fan_out')
        print('initialized Conv')

    elif isinstance(module, nn.RNNBase) or isinstance(module, nn.LSTMCell) or isinstance(module, nn.GRUCell):
        for name, param in module.named_parameters():
            if 'weight' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.normal_(param.data)

        print('initialized LSTM')

    elif isinstance(module, nn.BatchNorm1d) or isinstance(module, nn.BatchNorm2d):
        module.weight.data.normal_(1.0, 0.02)
        print('initialized BatchNorm')

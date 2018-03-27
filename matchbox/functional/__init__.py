import torch

from .nnet import dropout, linear, embedding, softmax, cross_entropy
from .elementwise import log, sqrt, sin, cos, tan, relu, tanh, sigmoid
from .reduction import any, all
from .tensor_math import matmul
from .indexing import getitem
from .tensor_shape import split, chunk, cat, stack, unbind, contiguous, view, transpose, permute, split_dim, join_dims, size_as_tensor, maxsize
from .special import causal_mask
from .constructors import new_zeros

import sys

# monkeypatching
# the giant hammer approach has problems:
# torch.nn.functional = sys.modules[__name__]
# so instead we'll do it piecemeal

import torch.nn.modules.sparse
import torch.nn.modules.linear
import torch.nn.modules.dropout
import torch.nn._functions.rnn

torch.nn.modules.sparse.F = sys.modules[__name__]
torch.nn.modules.linear.F = sys.modules[__name__]
torch.nn.modules.dropout.F = sys.modules[__name__]
torch.nn._functions.rnn.F = sys.modules[__name__]

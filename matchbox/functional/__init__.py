# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from .nnet import dropout, linear, embedding, softmax, cross_entropy
from .elementwise import log, sqrt, sin, cos, tan, relu, tanh, sigmoid
from .tensor_math import matmul
from .indexing import getitem
from .tensor_shape import split, chunk, cat, stack, unbind
from .tensor_shape import contiguous, view, transpose, permute
from .tensor_shape import split_dim, join_dims, size_as_tensor, maxsize
from .special import causal_mask
from . import reduction
from . import constructors

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

if torch.__version__ < '0.4':
    def embed_forward(self, input):
        return embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
    torch.nn.Embedding.forward = embed_forward

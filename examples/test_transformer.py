# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import argparse
import random
from collections import namedtuple

import torch
from torch import nn
from torch.autograd import Variable

import matchbox
from matchbox import MaskedBatch
from matchbox import functional as F
from matchbox.test_utils import mb_test, mb_rand, mb_assert
from matchbox.data import MaskedBatchField

from transformer import *

def test_LayerNorm():
    mb_test(LayerNorm(2),
            (4, (True, 3), (False, 2)))

def test_FeedForward():
    mb_test(FeedForward(2, 3, 0),
            (4, (True, 3), (False, 2)))

def test_ResidualBlock():
    mb_test(ResidualBlock(FeedForward(2, 3, 0), 2, 0),
            (4, (True, 3), (False, 2)))

def test_Attention():
    mb_test(Attention(2, 0, False),
            (4, (True, 3), (False, 2)), 0, 0)
    mb_test(Attention(2, 0, False),
            (4, (True, 3), (False, 2)), (4, (True, 3), (False, 2)), 1)
    mb_test(Attention(2, 0, True),
            (4, (True, 3), (False, 2)), 0, 0)

def test_MultiHead():
    mb_test(MultiHead(Attention(6, 0, False), 6, 6, 3),
            (4, (True, 3), (False, 6)), (4, (True, 3), (False, 6)), 1)
    mb_test(MultiHead(Attention(6, 0, True), 6, 6, 3),
            (4, (True, 3), (False, 6)), 0, 0)

def test_EncoderLayer():
    args = argparse.Namespace()
    args.__dict__.update(d_model=6, d_hidden=6, n_heads=3, drop_ratio=0)
    mb_test(EncoderLayer(args),
            (4, (True, 3), (False, 6)))

def test_DecoderLayer():
    args = argparse.Namespace()
    args.__dict__.update(d_model=6, d_hidden=6, n_heads=3, drop_ratio=0)
    mb_test(DecoderLayer(args),
            (4, (True, 3), (False, 6)), (4, (True, 3), (False, 6)))

def test_posenc():
    mb_test(lambda x: x + positional_encodings_like(x),
            (4, (True, 3), (False, 6)))

def test_Encoder():
    args = argparse.Namespace()
    args.__dict__.update(d_model=6, d_hidden=6, n_heads=3, drop_ratio=0,
                         n_layers=2)
    field = MaskedBatchField()
    field.out = nn.Linear(args.d_model, 5)
    xs = [Variable(torch.LongTensor(1, random.randint(1, 3)).random_(5))
          for i in range(4)]
    xb = MaskedBatch.fromlist(xs, (True,))
    mb_assert(Encoder(field, args),
              (xs,), (xb,), 4)

def test_Transformer():
    args = argparse.Namespace()
    B, T, C, V = 4, 3, 6, 5
    args.__dict__.update(d_model=C, d_hidden=C, n_heads=3, drop_ratio=0,
                         n_layers=2, length_ratio=1.5)
    field = MaskedBatchField()
    field.vocab = list(range(V))
    xs = [Variable(torch.LongTensor(1, random.randint(1, T)).random_(V))
          for i in range(B)]
    ys = [Variable(torch.LongTensor(1, random.randint(2, T)).random_(V))
          for i in range(B)]
    xb = MaskedBatch.fromlist(xs, (True,))
    yb = MaskedBatch.fromlist(ys, (True,))
    model = Transformer(field, field, args)
    mb_assert(model,
              (xs, ys), (xb, yb), B)
    def loss(x, y):
        b = namedtuple('_batch', ('src', 'trg'))(x, y)
        return model.loss(b, reduce=False)
    mb_assert(loss,
              (xs, ys), (xb, yb), B)

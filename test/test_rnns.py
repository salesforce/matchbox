# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch
from torch.autograd import Variable
from torch import nn
import matchbox
from matchbox import functional as F
from matchbox import MaskedBatch, batch
from matchbox.test_utils import mb_test, mb_assert

import random

def test_rnn_cell():
    mb_test(nn.RNNCell(2, 2), (4, (False, 2)), (4, (False, 2)))

@batch
def simple_rnn(x, h0, cell):
    h = h0
    for xt in x.unbind(1):
        h = cell(xt, h)
    return h

def test_rnn():
    def SimpleRNN(cell):
        def inner(x, h0):
            return simple_rnn(x, h0, cell)
        return inner
    mb_test(SimpleRNN(nn.RNNCell(2, 2)),
            (4, (True, 3), (False, 2)), (4, (False, 2)))

class RNNClass(nn.Module):
    def __init__(self, cell):
        super().__init__()
        self.cell = cell
    @batch
    def forward(self, x, h0=None):
        h = x.new(x.size(0), x.size(-1)).zero_() if h0 is None else h0
        for xt in x.unbind(1):
            h = self.cell(xt, h)
        return h

def test_rnn_class():
    mb_test(RNNClass(nn.RNNCell(2, 2)),
            (4, (True, 3), (False, 2)))

class LSTMClass(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.cell = nn.LSTMCell(in_size, out_size)
    @batch
    def forward(self, x, h0=None, c0=None):
        h = x.new(x.size(0), x.size(-1)).zero_() if h0 is None else h0
        c = x.new(x.size(0), x.size(-1)).zero_() if c0 is None else c0
        for xt in x.unbind(1):
            state = self.cell(xt, (h, c))
            h = state[0]
            c = state[1]
        return h

def test_lstm_class():
    mb_test(LSTMClass(2, 2),
            (4, (True, 3), (False, 2)))

class BiLSTMClass(nn.Module):
    def __init__(self, in_size, out_size):
        super().__init__()
        self.fcell = nn.LSTMCell(in_size, out_size)
        self.rcell = nn.LSTMCell(in_size, out_size)
    @batch
    def forward(self, x, h0=None, c0=None):
        hf = x.new(x.size(0), x.size(-1)).zero_() if h0 is None else h0
        cf = x.new(x.size(0), x.size(-1)).zero_() if c0 is None else c0
        for xt in x.unbind(1):
            state = self.fcell(xt, (hf, cf))
            hf = state[0]
            cf = state[1]
        hr = x.new(x.size(0), x.size(-1)).zero_() if h0 is None else h0
        cr = x.new(x.size(0), x.size(-1)).zero_() if c0 is None else c0
        for xt in reversed(x.unbind(1)):
            state = self.rcell(xt, (hr, cr))
            hr = state[0]
            cr = state[1]
        return hf, hr

def test_bilstm_class():
    mb_test(BiLSTMClass(2, 2),
            (4, (True, 3), (False, 2)))

class AccumRNNClass(nn.Module):
    def __init__(self, cell, dynamic):
        super().__init__()
        self.cell = cell
        self.dynamic = dynamic
    @batch
    def forward(self, x, h0=None):
        h = x.new(x.size(0), x.size(-1)).zero_() if h0 is None else h0
        encoding = []
        for xt in x.unbind(1):
            h = self.cell(xt, h)
            encoding.append(h)
        return F.stack(encoding, 1, self.dynamic)

def test_accum_rnn_class():
    mb_test(AccumRNNClass(nn.RNNCell(2, 2), None),
            (4, (True, 3), (False, 2)))
    mb_test(AccumRNNClass(nn.RNNCell(2, 2), True),
            (4, (True, 3), (False, 2)))

class AccumBiRNNClass(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.fwd = nn.RNNCell(size, size)
        self.bwd = nn.RNNCell(size, size)
    @batch
    def forward(self, x):
        h = h0 = x.batch_zeros(x.size(-1))
        fwd, bwd = [], []
        for xt in x.unbind(1):
            h = self.fwd(xt, h)
            fwd.append(h)
        fwd = F.stack(fwd, 1)
        h = h0
        for xt in reversed(x.unbind(1)):
            h = self.bwd(xt, h)
            bwd.append(h)
        bwd = F.stack(reversed(bwd), 1)
        return F.cat((fwd, bwd), 2)

def test_accum_birnn_class():
    mb_test(AccumBiRNNClass(1),
            (4, (True, 3), (False, 1)))

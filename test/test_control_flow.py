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

@batch
def while_loop(x):
    while x > 0:
        x = x - 1
    return x

def test_while():
    mb_test(while_loop, (4, ()))

@batch
def if_else(x):
    if x > 0:
        x = x - 1
    else:
        pass
    return x

def test_if_else():
    mb_test(if_else, (4, ()))

@batch
def if_noelse(x):
    if x > 0:
        x = x - 1
    return x

def test_if_noelse():
    mb_test(if_noelse, (4, ()))

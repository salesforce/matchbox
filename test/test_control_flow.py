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
from matchbox import MaskedBatch, batch, BD
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

@batch
def test_parse():
    SHIFT = 0
    REDUCE = 1

    class Queue:
        def __init__(self, transitions):
            self.q = transitions.same_zeros()
            self.push_ptr = transitions.batch_zeros()
            self.pop_ptr = transitions.batch_zeros()
        def push(self, x):
            self.q[BD, self.push_ptr] = x
            self.push_ptr += 1
        def pop(self):
            ret = self.q[BD, self.pop_ptr]
            self.pop_ptr += 1
            return ret

    def thin_stack(buffer, transitions):
        buffer_pointer = transitions.batch_zeros()
        stack = transitions.same_zeros().float().unsqueeze(2).expand(
            *transitions.maxsize(), buffer.size(2))
        ptr_queue = Queue(transitions)
        for t, transition in enumerate(transitions.unbind(1)):
            if transition == SHIFT:
                stack[BD, t] = buffer[BD, buffer_pointer]
                buffer_pointer = buffer_pointer + 1
            else:
                right = stack[BD, ptr_queue.pop()]
                left = stack[BD, ptr_queue.pop()]
                stack[BD, t] = left + right # standin for compose
            ptr_queue.push(t)
        return stack[BD, ptr_queue.pop()]

    buffer = matchbox.TENSOR_TYPE(torch.rand(1, 3, 2))
    transitions = matchbox.TENSOR_TYPE(torch.LongTensor([[0, 0, 1, 0, 1]]))
    print(thin_stack(buffer, transitions))

    buffer = matchbox.MaskedBatch.fromlist([buffer], (True, False))
    transitions = matchbox.MaskedBatch.fromlist([transitions], (True,))
    print(thin_stack(buffer, transitions))

    #mb_test(parse,
    #        (4, (True, 3)), (4, (True, 3), (False, 2)))

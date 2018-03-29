# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from . import functional

if torch.__version__ < '0.4':
    MAYBE_VARIABLE = TENSOR_TYPE = torch.autograd.Variable

    def _var_method(method_name):
        def inner(self, *args, **kwargs):
            t = getattr(self.data, method_name)(*args, **kwargs)
            return torch.autograd.Variable(t)
        return inner
    torch.autograd.Variable.new = _var_method('new')

    def _where(cond, x, y):
        cond = cond.type_as(x)
        return x * cond + y * (1 - cond)
    torch.where = _where

    _old_arange = torch.arange
    def _new_arange(*args, out=None):
        if isinstance(out, torch.autograd.Variable):
            torch.arange(*args, out=out.data)
            return out
        return _old_arange(*args, out=out)
    torch.arange = _new_arange

    def embed_forward(self, input):
        return functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
    torch.nn.Embedding.forward = embed_forward

else:
    def identity(x): return x
    MAYBE_VARIABLE = identity
    TENSOR_TYPE = torch.Tensor

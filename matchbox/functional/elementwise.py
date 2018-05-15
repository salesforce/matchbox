# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch
from torch.nn import functional as F

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def _elementwise_unary(fn):
    def inner(batch, *args, **kwargs):
        if not isinstance(batch, MaskedBatch):
            return fn(batch, *args, **kwargs)
        data = fn(batch.data, *args, **kwargs)
        mask = batch.mask.type_as(data)
        dims = batch.dims
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.float = _elementwise_unary(TENSOR_TYPE.float)
MaskedBatch.double = _elementwise_unary(TENSOR_TYPE.double)
MaskedBatch.byte = _elementwise_unary(TENSOR_TYPE.byte)
MaskedBatch.int = _elementwise_unary(TENSOR_TYPE.int)
MaskedBatch.long = _elementwise_unary(TENSOR_TYPE.long)

MaskedBatch.floor = _elementwise_unary(TENSOR_TYPE.floor)
MaskedBatch.ceil = _elementwise_unary(TENSOR_TYPE.ceil)
MaskedBatch.clamp = _elementwise_unary(TENSOR_TYPE.clamp)

MaskedBatch.log = log = _elementwise_unary(TENSOR_TYPE.log)
MaskedBatch.sqrt = sqrt = _elementwise_unary(TENSOR_TYPE.sqrt)
MaskedBatch.sin = sin = _elementwise_unary(TENSOR_TYPE.sin)
MaskedBatch.cos = cos = _elementwise_unary(TENSOR_TYPE.cos)
MaskedBatch.tan = tan = _elementwise_unary(TENSOR_TYPE.tan)

MaskedBatch.relu = relu = _elementwise_unary(F.relu)
MaskedBatch.tanh = tanh = _elementwise_unary(F.tanh)
MaskedBatch.sigmoid = sigmoid = _elementwise_unary(F.sigmoid)

def _elementwise_binary(fn):
    def inner(batch1, batch2, **kwargs):
        if not isinstance(batch1, MaskedBatch) and not isinstance(batch2, MaskedBatch):
            return fn(batch1, batch2, **kwargs)
        if isinstance(batch2, MaskedBatch):
            data = fn(batch1.data, batch2.data, **kwargs)
            mask = batch1.mask * batch2.mask
            dims = tuple(b1 or b2 for b1, b2 in zip(batch1.dims, batch2.dims))
        else:
            data = fn(batch1.data, batch2, **kwargs)
            mask = batch1.mask.type_as(data)
            dims = batch1.dims
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.__neg__ = _elementwise_binary(TENSOR_TYPE.__neg__)
MaskedBatch.__add__ = _elementwise_binary(TENSOR_TYPE.__add__)
MaskedBatch.__sub__ = _elementwise_binary(TENSOR_TYPE.__sub__)
MaskedBatch.__mul__ = _elementwise_binary(TENSOR_TYPE.__mul__)
MaskedBatch.__truediv__ = _elementwise_binary(TENSOR_TYPE.__truediv__)
MaskedBatch.__radd__ = _elementwise_binary(TENSOR_TYPE.__radd__)
MaskedBatch.__rsub__ = _elementwise_binary(TENSOR_TYPE.__rsub__)
MaskedBatch.__rmul__ = _elementwise_binary(TENSOR_TYPE.__rmul__)
MaskedBatch.__rtruediv__ = _elementwise_binary(TENSOR_TYPE.__rtruediv__)

MaskedBatch.__lt__ = _elementwise_binary(TENSOR_TYPE.__lt__)
MaskedBatch.__le__ = _elementwise_binary(TENSOR_TYPE.__le__)
MaskedBatch.__eq__ = _elementwise_binary(TENSOR_TYPE.__eq__)
MaskedBatch.__ne__ = _elementwise_binary(TENSOR_TYPE.__ne__)
MaskedBatch.__gt__ = _elementwise_binary(TENSOR_TYPE.__gt__)
MaskedBatch.__ge__ = _elementwise_binary(TENSOR_TYPE.__ge__)

MaskedBatch.lt = _elementwise_binary(TENSOR_TYPE.lt)
MaskedBatch.le = _elementwise_binary(TENSOR_TYPE.le)
MaskedBatch.eq = _elementwise_binary(TENSOR_TYPE.eq)
MaskedBatch.ne = _elementwise_binary(TENSOR_TYPE.ne)
MaskedBatch.gt = _elementwise_binary(TENSOR_TYPE.gt)
MaskedBatch.ge = _elementwise_binary(TENSOR_TYPE.ge)

def _inject_arith(original, replacement):
    def inner(self, other):
        if isinstance(other, MaskedBatch):
            return replacement(self, other)
        return original(self, other)
    return inner

TENSOR_TYPE.__add__ = _inject_arith(TENSOR_TYPE.__add__, lambda a, b: b + a)
TENSOR_TYPE.__sub__ = _inject_arith(TENSOR_TYPE.__sub__, lambda a, b: -b + a)
TENSOR_TYPE.__mul__ = _inject_arith(TENSOR_TYPE.__mul__, lambda a, b: b * a)
# TODO fix __sub__; it's ugly
# TENSOR_TYPE.__matmul__ = _inject_arith(TENSOR_TYPE.__matmul__, lambda a, b:)
# TENSOR_TYPE.__truediv__ = _inject_arith(TENSOR_TYPE.__truediv__, lambda a, b:)

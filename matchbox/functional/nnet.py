# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch
from torch.nn import functional as F

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def dropout(batch, p=0.5, training=False, inplace=False):
    if not isinstance(batch, MaskedBatch):
        return F.dropout(batch, p, training, inplace)
    data = F.dropout(batch.data, p, training, inplace)
    return MaskedBatch(data, batch.mask, batch.dims)

MaskedBatch.dropout = dropout
TENSOR_TYPE.dropout = dropout

def linear(batch, weight, bias=None):
    if not isinstance(batch, MaskedBatch):
        return F.linear(batch, weight, bias)
    if batch.dims[-1]:
        raise ValueError("cannot contract static and dynamic dimensions")
    data = F.linear(batch.data, weight, bias)
    return MaskedBatch(data, batch.mask, batch.dims)

def embedding(batch, weight, padding_idx=None, max_norm=None, norm_type=2,
              scale_grad_by_freq=False, sparse=False):
    def compat_embedding(batch, weight, padding_idx, max_norm, norm_type,
                         scale_grad_by_freq, sparse):
        if torch.__version__ >= '0.4':
            return F.embedding(batch, weight, padding_idx, max_norm, norm_type,
                               scale_grad_by_freq, sparse)
        if padding_idx is not None:
            raise ValueError("F.embedding doesn't support padding_idx for torch < 0.4")
        return F.embedding(batch, weight, max_norm, norm_type,
                           scale_grad_by_freq, sparse)

    if not isinstance(batch, MaskedBatch):
        return compat_embedding(batch, weight, padding_idx, max_norm, norm_type,
                                scale_grad_by_freq, sparse)
    #data = batch.data - batch.mask
    data = batch.data
    data = compat_embedding(
        data, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    mask = batch.mask.unsqueeze(-1).float()
    dims = batch.dims + (False,)
    return MaskedBatch(data, mask, dims)

def softmax(batch, dim=-1):
    if not isinstance(batch, MaskedBatch):
        return F.softmax(batch, dim)
    if dim == 0:
        raise ValueError("cannot softmax over batch dimension")
    elif dim < 0:
        dim += batch.dim()
    dims = batch.dims
    if dims[dim - 1]:
        data = F.softmax(batch.data * batch.mask, dim) * batch.mask
        data = data / data.sum(dim, keepdim=True)
        data[data.ne(data).detach()] = 0 # remove NaNs
        mask = batch.mask.narrow(dim, 0, 1)
        dims = dims[:dim - 1] + (False,) + dims[dim:]
    else:
        data = F.softmax(batch.data, dim)
        mask = batch.mask
    return MaskedBatch(data, mask, dims)

MaskedBatch.softmax = softmax
TENSOR_TYPE.softmax = softmax

def cross_entropy(input, target, weight=None, size_average=True,
                  ignore_index=-1, reduce=True):
    if not isinstance(input, MaskedBatch) and not isinstance(target, MaskedBatch):
        ret = F.cross_entropy(input.contiguous().view(-1, input.size(-1)),
                              target.contiguous().view(-1),
                              weight, size_average, ignore_index, reduce)
        if reduce: return ret
        return ret.view(input.size(0), input.size(1))
    target_data = (target.data + target.mask - 1).view(-1)
    input_data = input.data.view(target_data.size(0), -1)
    if ignore_index != -1:
        raise ValueError("cannot set ignore_index with MaskedBatch")
    data = F.cross_entropy(
        input_data, target_data, weight, size_average, ignore_index, reduce)
    if reduce: return data
    data = data.view(input.maxsize(0), input.maxsize(1))
    mask = input.mask.squeeze(-1) * target.mask.float()
    return MaskedBatch(data, mask, target.dims)

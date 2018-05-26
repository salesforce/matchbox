# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def _inject_new(original):
    def inner(self, *sizes, **kwargs):
        source = self.data if isinstance(self, MaskedBatch) else self
        if not any(isinstance(size, MaskedBatch) for size in sizes):
            return original(source, *(int(size) for size in sizes), **kwargs)
        if isinstance(sizes[0], MaskedBatch):
            raise ValueError("batch size dimension must be static")
        dims = tuple(isinstance(size, MaskedBatch) for size in sizes[1:])
        maxsizes = [size.data.max() if isinstance(size, MaskedBatch)
                    else int(size) for size in sizes]
        bs = maxsizes[0]
        masksizes = [s if b else 1 for s, b in zip(maxsizes[1:], dims)]
        data = original(source, *maxsizes, **kwargs)
        mask = source.new_zeros(bs, *masksizes, **kwargs)
        # TODO this should be
        # mask[range(bs), *(s - 1 for s in masksizes)] = 1
        # mask = mask[:, *(slice(None, None, -1) if b
        #                  else slice(None, None, None) for b in dims)]
        # for d, b in enumerate(dims):
        #     if not b: continue
        #     mask = mask.cumsum(d + 1)
        # mask = mask[:, *(slice(None, None, -1) if b
        #                  else slice(None, None, None) for b in dims)]
        # if faking negative strides is fast enough;
        # we can also use numpy if it's worth it.
        for i in range(bs):
            inds = [slice(0, int(size.data[i])) if b else slice(None)
                    for size, b in zip(sizes[1:], dims)]
            mask[(slice(i, i + 1), *inds)] = 1
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.new_empty = TENSOR_TYPE.new_empty = _inject_new(
    TENSOR_TYPE.new_empty)
MaskedBatch.new_zeros = TENSOR_TYPE.new_zeros = _inject_new(
    TENSOR_TYPE.new_zeros)
MaskedBatch.new_ones = TENSOR_TYPE.new_ones = _inject_new(
    TENSOR_TYPE.new_ones)

def _inject_batch_new(original):
    def inner(batch, *sizes, **kwargs):
        # if len(sizes) == 0:
        #     return original(batch, *batch.size(), **kwargs)
        return original(batch, batch.size(0), *sizes, **kwargs)
    return inner

MaskedBatch.batch_empty = TENSOR_TYPE.batch_empty = _inject_batch_new(
    TENSOR_TYPE.new_empty)
MaskedBatch.batch_zeros = TENSOR_TYPE.batch_zeros = _inject_batch_new(
    TENSOR_TYPE.new_zeros)
MaskedBatch.batch_ones = TENSOR_TYPE.batch_ones = _inject_batch_new(
    TENSOR_TYPE.new_ones)

def _inject_same(original):
    def inner(batch):
        if not isinstance(batch, MaskedBatch):
            return batch.new_zeros(*batch.size())
        data = batch.new_zeros(*batch.maxsize())
        return MaskedBatch(data, batch.mask, batch.dims)
    return inner

MaskedBatch.same_empty = TENSOR_TYPE.same_empty = _inject_same(
    TENSOR_TYPE.new_empty)
MaskedBatch.same_zeros = TENSOR_TYPE.same_zeros = _inject_same(
    TENSOR_TYPE.new_zeros)
MaskedBatch.same_ones = TENSOR_TYPE.same_ones = _inject_same(
    TENSOR_TYPE.new_ones)

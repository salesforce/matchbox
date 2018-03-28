# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def _reduce(fn, zero_preserving=False):
    def inner(batch, dim=None, keepdim=False):
        if dim is None:
            if not zero_preserving and __builtins__['any'](batch.dims):
                raise NotImplementedError(
                    "cannot reduce to scalar with non-zero-preserving kernel "
                    "if dynamic dims present")
            mask = batch.mask[(slice(None), *(0 for d in batch.dims))]
            dims = ()
        else:
            if dim < 0:
                dim += batch.dim()
            if not zero_preserving and batch.dims[dim - 1]:
                raise NotImplementedError("cannot reduce over dynamic dim "
                                          "with non-zero-preserving kernel")
            if keepdim:
                mask = batch.mask[tuple(slice(0, 1) if i == dim else slice(None)
                                        for i in range(batch.mask.dim()))]
                dims = tuple(False if i == dim - 1 else d
                             for i, d in enumerate(batch.dims))
            else:
                mask = batch.mask[tuple(0 if i == dim else slice(None)
                                        for i in range(batch.mask.dim()))]
                dims = tuple(d for i, d in enumerate(batch.dims)
                             if i != dim - 1)
        data = fn(batch.data * batch.mask, dim=dim, keepdim=keepdim)
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.sum = _reduce(torch.sum, zero_preserving=True)
MaskedBatch.mean = _reduce(torch.mean)
MaskedBatch.std = _reduce(torch.std)

def any(batch):
    return (batch.data * batch.mask).any()

MaskedBatch.any = any

def all(batch):
    return (batch.data * batch.mask).all()

MaskedBatch.all = all

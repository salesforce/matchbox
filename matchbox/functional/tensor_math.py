# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def matmul(batch1, batch2):
    if not isinstance(batch1, MaskedBatch) and not isinstance(batch2, MaskedBatch):
        return batch1 @ batch2
    if isinstance(batch1, MaskedBatch) and isinstance(batch2, MaskedBatch):
        dims1 = len(batch1.dims)
        dims2 = len(batch2.dims)
        data1 = batch1.data * batch1.mask
        data2 = batch2.data * batch2.mask
        if dims1 == 1:
            data1 = data1.unsqueeze(-2)
        if dims2 == 1 and dims1 == 1:
            data2 = data2.unsqueeze(-1)
        data = data1 @ data2
        if dims1 == 1 and dims2 == 1:
            #if (batch1.dims[0] or batch2.dims[0]) and not batch1.mask.eq(batch2.mask).all():
            #    raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :1]
            dims = ()
        if dims1 == 2 and dims2 == 1:
            #if (batch1.dims[1] or batch2.dims[0]) and not batch1.mask[:, 0].eq(batch2.mask).all():
            #    raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :, :1] @ batch2.mask[:, :1]
            dims = batch1.dims[:1]
        elif dims1 == 1 and dims2 == 2:
            #if (batch1.dims[0] or batch2.dims[0]) and not batch1.mask.eq(batch2.mask[:, :, 0]).all():
            #    raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :1].unsqueeze(-2) @ batch2.mask[:, :1, :]
            dims = batch2.dims[1:]
        elif dims1 == 2 and dims2 == 2:
            #if (batch1.dims[1] or batch2.dims[0]) and not batch1.mask[:, 0].eq(batch2.mask[:, :, 0]).all():
            #    raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :, :1] @ batch2.mask[:, :1, :]
            dims = batch1.dims[:1] + batch2.dims[1:]
        else:
            raise NotImplementedError("matmul not implemented with batches of 3+D tensors")
    else:
        raise NotImplementedError("matmul not implemented between MaskedBatch and tensor")
    return MaskedBatch(data, mask, dims)

MaskedBatch.__matmul__ = matmul

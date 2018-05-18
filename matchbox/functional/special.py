# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def causal_mask(batch, in_dim, out_dim):
    '''if in_dim is indexed by i and out_dim by j, masks ret[i,j] where i > j'''
    if not isinstance(batch, MaskedBatch):
        # TODO or we could just promote to MaskedBatch /shrug
        if in_dim == 1 and out_dim == 2:
            return batch - batch.new(
                *batch.size()[1:]).fill_(1e10).tril(-1).unsqueeze(0)
        elif in_dim == 2 and out_dim == 1:
            return batch - batch.new(
                *batch.size()[1:]).fill_(1e10).triu(1).unsqueeze(0)
        else:
            raise NotImplementedError("unsupported arguments for causal_mask")
    if in_dim == 1 and out_dim == 2:
        mask = batch.mask * batch.mask.new(
            *batch.data.size()[1:]).fill_(1).triu(0).unsqueeze(0)
    elif in_dim == 2 and out_dim == 1:
        mask = batch.mask * batch.mask.new(
            *batch.data.size()[1:]).fill_(1).tril(0).unsqueeze(0)
    else:
        raise NotImplementedError("unsupported arguments for causal_mask")
    dims = tuple(True if d + 1 in (in_dim, out_dim) else b
                 for d, b in enumerate(batch.dims))
    return MaskedBatch(batch.data, mask, dims)

MaskedBatch.causal_mask = causal_mask
TENSOR_TYPE.causal_mask = causal_mask

def _synchronize(batch):
    if not isinstance(batch, MaskedBatch):
        return batch
    if any(batch.dims):
        raise ValueError("cannot synchronize batch with dynamic dimensions")
    mask = batch.mask + (1 - batch.mask)
    return MaskedBatch(batch.data, mask, batch.dims)

MaskedBatch._synchronize = _synchronize
TENSOR_TYPE._synchronize = _synchronize

def _update(batch, new, update_mask=None):
    if not isinstance(new, MaskedBatch) and (
            not isinstance(batch, MaskedBatch) or update_mask is None):
        return new
    update_mask = (new.mask.byte() if update_mask is None else
                   (update_mask.data * update_mask.mask).expand_as(new.mask))
    if isinstance(batch, MaskedBatch):
        data = torch.where(update_mask, new.data, batch.data)
    else:
        data = torch.where(update_mask, new.data, batch)
    return MaskedBatch(data, update_mask.type_as(data), new.dims)

MaskedBatch._update = _update
TENSOR_TYPE._update = _update

# def _for(closure, iterator):
#     for i in iterator:
#         closure(i)

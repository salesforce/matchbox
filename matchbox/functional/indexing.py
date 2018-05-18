# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch, BD
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def _check_index(batch, index):
    if not isinstance(index, tuple) or index[0] is not BD:
        return index, None
    # TODO may be able to use _check_advanced_indexing
    # and/or _preprocess_adv_index_seq
    if any(map(lambda i: isinstance(i, (MaskedBatch, TENSOR_TYPE, list, tuple)),
               index[1:])):
        # advanced indexing case
        return (torch.arange(batch.size(0)).long(),) + index[1:], True
    if any(map(lambda i: i is None, index)):
        return (slice(None),) + index[1:], None
    # basic slicing case
    return (slice(None),) + index[1:], False

def _fix_negative_mask_slices(batch, index):
    index = list(index)
    for i, (ind, b) in enumerate(zip(index[1:], batch.dims)):
        if b:
            if isinstance(ind, int) and ind < 0:
                raise NotImplementedError("cannot index dynamic dim with "
                                          "negative integer")
            if (isinstance(ind, slice) and ind.stop is not None
                    and ind.stop < 0):
                if ind.step is not None or ind.start is not None:
                    raise NotImplementedError("cannot index dynamic dim "
                                              "with complex slice")
                index[i + 1] = slice(-ind.stop, None)
    return tuple(index)

def _wrap_getitem(original):
    def getitem(batch, index):
        index, advanced = _check_index(batch, index)
        if not isinstance(batch, MaskedBatch):
            return original(batch, index)
        if advanced is None:
            raise NotImplementedError("unimplemented indexing pattern")
        indexdata = tuple(i.data if isinstance(i, MaskedBatch)
                          else i for i in index)
        indexmask = tuple(i.mask if isinstance(i, MaskedBatch)
                          else slice(None) for i in index)
        data = batch.data[indexdata]
        indexdata = _fix_negative_mask_slices(batch, indexdata)
        mask = batch.mask.same_zeros()
        mask[indexmask] = 1
        mask = batch.mask * mask
        mask = mask[tuple(i if b else 0 if isinstance(i, int)
                          else slice(None) for i, b in zip(
                            indexdata, (True,) + batch.dims))]
        dims = tuple(b for i, b in zip(
            index[1:] + (slice(None),) * len(batch.dims), batch.dims)
                     if isinstance(i, slice)) # could be faster
        return MaskedBatch(data, mask, dims)
    return getitem

MaskedBatch.__getitem__ = TENSOR_TYPE.__getitem__ = _wrap_getitem(
    TENSOR_TYPE.__getitem__)

def _wrap_setitem(original):
    def setitem(batch, index, value):
        index, advanced = _check_index(batch, index)
        if not isinstance(batch, MaskedBatch):
            original(batch, index, value)
            return
        if advanced is None:
            raise NotImplementedError("unimplemented indexing pattern")
        if not isinstance(value, MaskedBatch):
            batch.data[index] = value
            return batch # TODO should MaskedBatch be mutable?
            # maybe a rule is to allow mutating _only_ the data
        data = batch.data.clone() # TODO make in-place
        indexdata = tuple(i.data if isinstance(i, MaskedBatch)
                          else i for i in index)
        indexmask = tuple(i.mask if isinstance(i, MaskedBatch)
                          else slice(None) for i in index)
        data[indexdata] = value.data
        batchmask = batch.mask.clone()
        mask = value.mask.same_zeros()
        mask[indexmask] = 1
        batchmask[indexdata] = value.mask * mask
        data = torch.where(batchmask.byte(), data, batch.data)
        return MaskedBatch(data, batch.mask, batch.dims)
    return setitem

MaskedBatch.__setitem__ = TENSOR_TYPE.__setitem__ = _wrap_setitem(
    TENSOR_TYPE.__setitem__)

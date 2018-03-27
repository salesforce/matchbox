import torch
from torch.nn import functional as F

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def _inject_new(original):
    def inner(self, *sizes):
        source = self.data if isinstance(self, MaskedBatch) else self
        if not __builtins__['any'](isinstance(size, MaskedBatch)
                                   for size in sizes):
            return original(source, *(int(size) for size in sizes))
        if isinstance(sizes[0], MaskedBatch):
            raise ValueError("batch size dimension must be static")
        dims = tuple(isinstance(size, MaskedBatch) for size in sizes[1:])
        maxsizes = [size.data.max() if isinstance(size, MaskedBatch)
                    else int(size) for size in sizes]
        bs = maxsizes[0]
        masksizes = [s if b else 1 for s, b in zip(maxsizes[1:], dims)]
        data = original(source, *maxsizes)
        mask = source.new_zeros(bs, *masksizes)
        for i in range(bs):
            # TODO this is pretty terrible
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

def new_zeros(self, *sizes):
    return self.data.new_zeros(*sizes)
    # mask = batch.mask + (1 - batch.mask)
    # return MaskedBatch()

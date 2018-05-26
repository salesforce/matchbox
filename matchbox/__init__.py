# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from .compat import TENSOR_TYPE

# indexing sentinel for batch dimension
# required because the semantics we want is equivalent to basic indexing
# with : but advanced indexing with a range over the dimension.
BD = ()

class MaskedBatch(object):

    def __init__(self, data, mask, dims):
        if data.dim() != mask.dim() or mask.dim() != len(dims) + 1:
            raise ValueError("malformed MaskedBatch {} with:\n data: "
                             " {}\n mask: {}".format(
                repr(dims), repr(data), repr(mask)))
        if isinstance(mask, TENSOR_TYPE) and mask.requires_grad:
            raise ValueError("mask cannot require grad")
        self.data = data
        self.mask = mask
        self.dims = dims

    @classmethod
    def fromlist(cls, examples, dims):
        # TODO do some validation
        bs = len(examples)
        sizes = [max(x.size(d + 1) for x in examples)
                 for d in range(len(dims))]
        data = examples[0].new(bs, *sizes).zero_()
        mask_sizes = [s if dims[d] else 1 for d, s in enumerate(sizes)]
        mask = examples[0].new(bs, *mask_sizes).zero_()
        mask.requires_grad = False
        for i, x in enumerate(examples):
            inds = [slice(0, x.size(d + 1)) if b else slice(None)
                    for d, b in enumerate(dims)]
            data[(slice(i, i + 1), *inds)] = x
            mask[(slice(i, i + 1), *inds)] = 1
        return cls(data, mask, dims)

    def examples(self):
        data, mask, dims = self.data, self.mask.data.long(), self.dims
        for i in range(data.maxsize(0)):
            inds = tuple(slice(0, mask[i].sum(d, keepdim=True)[
                tuple(0 for _ in dims)])
                if b else slice(None) for d, b in enumerate(dims))
            yield data[(slice(i, i + 1), *inds)]

    def __repr__(self):
        return "MaskedBatch {} with:\n data: {}\n mask: {}".format(
            repr(self.dims), repr(self.data), repr(self.mask))

    def cuda(self, *args, **kwargs):
        data = self.data.cuda(*args, **kwargs)
        mask = self.mask.cuda(*args, **kwargs)
        return self.__class__(data, mask, self.dims)

    @property
    def is_cuda(self):
        return self.data.is_cuda

    def get_device(self):
        return self.data.get_device()

    def dim(self):
        return self.data.dim()

    def size(self, dim=None):
        if dim is None:
            if any(self.dims):
                raise ValueError("use size_as_tensor for dynamic dimensions")
            return self.data.size()
        if dim < 0:
            dim += self.dim()
        if dim == 0 or not self.dims[dim - 1]:
            return self.data.size(dim)
        raise ValueError("use size_as_tensor for dynamic dimensions")

    @property
    def shape(self):
        return self.size()

    def new(self, *sizes):
        return self.data.new(*sizes)

    def type(self, dtype=None, non_blocking=False, **kwargs):
        if dtype:
            data = self.data.type(dtype, non_blocking, **kwargs)
            mask = self.mask.type(dtype, non_blocking, **kwargs)
            return self.__class__(data, mask, self.dims)
        else:
            return self.data.type()

    def __bool__(self):
        if self.data.nelement() > 1:
            raise ValueError("bool value of MaskedBatch with more than one "
                             "value is ambiguous; use .any() or .all() or wrap "
                             "code containing control flow in @batch.")
        return bool(self.data)

from . import functional
from .macro import batch

try:
    from . import data
except ImportError:
    pass

# global mask stack for control flow; not thread-safe
_EXECUTION_MASKS = [None]
EXECUTION_MASK = None

def push_execution_mask(mask):
    global EXECUTION_MASK
    if EXECUTION_MASK is not None:
        EXECUTION_MASK = EXECUTION_MASK * mask
    else:
        EXECUTION_MASK = mask
    _EXECUTION_MASKS.append(EXECUTION_MASK)
    EXECUTION_MASK = mask

def pop_execution_mask():
    global EXECUTION_MASK
    _EXECUTION_MASKS.pop()
    EXECUTION_MASK = _EXECUTION_MASKS[-1]

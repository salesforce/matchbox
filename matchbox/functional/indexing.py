# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def getitem(batch, index):
    if not isinstance(index, tuple) or index[0] != slice(None):
        raise ValueError("first index must be :")
    if None in index:
        raise NotImplementedError("cannot index with None")
    data = batch.data[index]
    index = list(index)
    for i, (ind, b) in enumerate(zip(index[1:], batch.dims)):
        if b:
            if isinstance(ind, int) and ind < 0:
                raise NotImplementedError("cannot index dynamic dim with "
                                          "negative integer")
            if isinstance(ind, slice) and ind.stop is not None and ind.stop < 0:
                if ind.step is not None or ind.start is not None:
                    raise NotImplementedError("cannot index dynamic dim with "
                                              "complex slice")
                index[i + 1] = slice(-ind.stop, None)
    index = tuple(index)
    mask = batch.mask[tuple(i if b else 0 if isinstance(i, int) else slice(None)
                       for i, b in zip(index, (True,) + batch.dims))]
    dims = tuple(b for i, b in zip(index[1:] + (slice(None),) * len(batch.dims),
                                   batch.dims)
                 if not isinstance(i, int)) # could be faster
    return MaskedBatch(data, mask, dims)

MaskedBatch.__getitem__ = getitem

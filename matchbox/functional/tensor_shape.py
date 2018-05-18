# Copyright (c) 2018, salesforce.com, inc.
# All rights reserved.
# Licensed under the BSD 3-Clause license.
# For full license text, see the LICENSE file in the repo root
# or https://opensource.org/licenses/BSD-3-Clause

import torch

from matchbox import MaskedBatch
from matchbox.compat import MAYBE_VARIABLE, TENSOR_TYPE

def split(batch, split_size_or_sections, dim=0):
    if not isinstance(batch, MaskedBatch):
        return torch.split(batch, split_size_or_sections, dim)
    if dim < 0:
        dim += batch.dim()
    if dim > 0 and batch.dims[dim - 1]:
        return tuple(MaskedBatch(data, mask, batch.dims) for data, mask in zip(
            torch.split(batch.data, split_size_or_sections, dim),
            torch.split(batch.mask, split_size_or_sections, dim)))
    return tuple(MaskedBatch(data, batch.mask, batch.dims) for data
                 in torch.split(batch.data, split_size_or_sections, dim))

MaskedBatch.split = split

def chunk(batch, chunks, dim=0):
    if dim < 0:
        dim += batch.dim()
    split_size = (batch.maxsize(dim) + chunks - 1) // chunks
    return split(batch, split_size, dim)

MaskedBatch.chunk = chunk

def cat(sequence, dim):
    sequence = list(sequence)
    if len(sequence) == 0:
        raise ValueError("cannot stack empty sequence")
    first = sequence[0]
    if not isinstance(first, MaskedBatch):
        return torch.cat(sequence, dim)
    data = torch.cat([batch.data for batch in sequence], dim)
    if first.dims[dim - 1]:
        mask = torch.cat([batch.mask for batch in sequence], dim)
    else:
        mask = first.mask
    return MaskedBatch(data, mask, first.dims)

def stack(sequence, dim, dynamic=None):
    sequence = list(sequence)
    if len(sequence) == 0:
        raise ValueError("cannot stack empty sequence")
    first = sequence[0]
    if not isinstance(first, MaskedBatch):
        return torch.stack(sequence, dim)
    if dim < 0:
        dim += first.dim() + 1
    if dynamic is None:
        dynamic = not first.mask.eq(sequence[-1].mask).all()
    data = torch.cat([batch.data.unsqueeze(dim) for batch in sequence], dim)
    if dynamic:
        mask = torch.cat(
            [batch.mask.unsqueeze(dim) for batch in sequence], dim)
    else:
        mask = first.mask.unsqueeze(dim)
    dims = first.dims[:dim - 1] + (dynamic,) + first.dims[dim - 1:]
    return MaskedBatch(data, mask, dims)

def unbind(batch, dim):
    if not isinstance(batch, MaskedBatch):
        return torch.unbind(batch, dim)
    if dim == 0:
        raise ValueError("cannot unbind over batch dimension")
    dims = tuple(b for d, b in enumerate(batch.dims) if d != dim - 1)
    if batch.dims[dim - 1]:
        return tuple(MaskedBatch(data, mask, dims)
                     for data, mask in zip(torch.unbind(batch.data, dim),
                                           torch.unbind(batch.mask, dim)))
    else:
        mask = batch.mask.squeeze(dim)
        return tuple(MaskedBatch(data, mask, dims)
                     for data in torch.unbind(batch.data, dim))

MaskedBatch.unbind = unbind
TENSOR_TYPE.unbind = unbind

def contiguous(batch):
    return MaskedBatch(
        batch.data.contiguous(), batch.mask.contiguous(), batch.dims)

MaskedBatch.contiguous = contiguous

def view(batch, *sizes):
    bs = batch.data.size(0)
    if sizes[0] not in (1, -1, bs):
        raise ValueError("first dim in view must be 1, -1, or batch size")
    sizes = (bs,) + sizes[1:]
    data = batch.data.view(*sizes) # TODO can throw
    mask_sizes = (bs,) + tuple(batch.data.size(i) if sizes[i] == -1 else 1
                               for i in range(1, len(sizes)))
    mask = batch.mask.view(*mask_sizes) # TODO can this throw if data doesn't?
    dims = tuple(sizes[i] == -1 for i in range(1, len(sizes)))
    return MaskedBatch(data, mask, dims)

MaskedBatch.view = view

def transpose(batch, dim1, dim2):
    if dim1 > batch.dim() or dim2 > batch.dim():
        if dim1 < 0:
            dim1 += batch.dim()
        if dim2 < 0:
            dim2 += batch.dim()
        permutation = [dim2 if i == dim1 else dim1 if i == dim2 else i
                       for i in range(batch.dim() + 1)][:batch.dim()]
        return batch.permute(*permutation)
    if not isinstance(batch, MaskedBatch):
        return torch.transpose(batch, dim1, dim2)
    data = batch.data.transpose(dim1, dim2)
    mask = batch.mask.transpose(dim1, dim2)
    dims = list(batch.dims)
    dims[dim1 - 1], dims[dim2 - 1] = dims[dim2 - 1], dims[dim1 - 1]
    dims = tuple(dims)
    return MaskedBatch(data, mask, dims)

MaskedBatch.transpose = transpose
TENSOR_TYPE.transpose = transpose

def permute(batch, *permutation):
    data = batch.data.permute(*permutation)
    mask = batch.mask.permute(*permutation)
    dims = tuple(batch.dims[i - 1] for i in permutation[1:])
    return MaskedBatch(data, mask, dims)

MaskedBatch.permute = permute

def squeeze(batch, dim):
    if dim < 0:
        dim += batch.dim()
    data = batch.data.squeeze(dim)
    mask = batch.mask.squeeze(dim)
    dims = batch.dims[:dim - 1] + batch.dims[dim:]
    return MaskedBatch(data, mask, dims)

MaskedBatch.squeeze = squeeze

def unsqueeze(batch, dim):
    if dim < 0:
        dim += batch.dim()
    data = batch.data.unsqueeze(dim)
    mask = batch.mask.unsqueeze(dim)
    dims = batch.dims[:dim - 1] + (False,) + batch.dims[dim - 1:]
    return MaskedBatch(data, mask, dims)

MaskedBatch.unsqueeze = unsqueeze

def expand(batch, *sizes):
    data = batch.data.expand(*sizes)
    return MaskedBatch(data, batch.mask, batch.dims)

MaskedBatch.expand = expand

def split_dim(batch, dim, split_by):
    if dim < 0:
        dim += batch.dim()
    if batch.data.size(dim) % split_by != 0:
        raise ValueError("size of dim not divisible by split_by")
    sizes = ((s // split_by, split_by) if d == dim else (s,)
             for d, s in enumerate(batch.data.size()))
    if not isinstance(batch, MaskedBatch):
        return batch.contiguous().view(*(n for tup in sizes for n in tup))
    if dim == 0:
        msizes = ((s // split_by, split_by) if d == dim else (s,)
                 for d, s in enumerate(batch.mask.size()))
        mask = batch.mask.contiguous().view(*(n for tup in msizes for n in tup))
        mask = mask.narrow(1, 0, 1)
    else:
        if batch.dims[dim - 1]:
            raise ValueError("cannot split dynamic dimension")
        mask = batch.mask.unsqueeze(dim)
    data = batch.data.contiguous().view(*(n for tup in sizes for n in tup))
    dims = batch.dims[:dim] + (False,) + batch.dims[dim:]
    return MaskedBatch(data, mask, dims)

MaskedBatch.split_dim = split_dim
TENSOR_TYPE.split_dim = split_dim

def join_dims(batch, dim1, dim2):
    if dim1 < 0:
        dim1 += batch.dim()
    if dim2 < 0:
        dim2 += batch.dim()
    if dim2 != dim1 + 1:
        order = [n for n in range(batch.dim()) if n != dim2]
        order.insert(dim1 + 1, dim2)
        batch = batch.permute(*order)
        if dim2 < dim1:
            dim1 -= 1
    if not isinstance(batch, MaskedBatch):
        sizes = (batch.size(d + 1) * s if d == dim1 else s
                 for d, s in enumerate(batch.size()) if d != dim1 + 1)
        return batch.contiguous().view(*sizes)
    sizes = (batch.data.size(d + 1) * s if d == dim1 else s
             for d, s in enumerate(batch.data.size()) if d != dim1 + 1)
    data = batch.data.contiguous().view(*sizes)
    if dim1 == 0:
        mask = batch.mask.expand(*(s if d == dim1 + 1 else -1
                                   for d, s in enumerate(batch.data.size())))
        sizes = (s * mask.size(d + 1) if d == dim1 else s
                 for d, s in enumerate(mask.size()) if d != dim1 + 1)
        mask = mask.contiguous().view(*sizes)
    else:
        mask = batch.mask.squeeze(dim1 + 1)
    dims = batch.dims[:dim1] + batch.dims[dim1 + 1:]
    return MaskedBatch(data, mask, dims)

MaskedBatch.join_dims = join_dims
TENSOR_TYPE.join_dims = join_dims

def size_as_tensor(batch, dim):
    if not isinstance(batch, MaskedBatch):
        return MAYBE_VARIABLE(torch.LongTensor([batch.size(dim)]))
    if dim is None:
        return tuple(batch.size(d) for d in range(len(batch.dims) + 1))
    if dim < 0:
        dim += batch.dim()
    if dim == 0 or not batch.dims[dim - 1]:
        return MAYBE_VARIABLE(torch.LongTensor([batch.data.size(dim)]))
    if any(batch.dims[:dim - 1] + batch.dims[dim:]):
        raise NotImplementedError("cannot get size in any of two or "
                                  "more dynamic dimensions")
    data = batch.mask.long().sum(dim).view(-1)
    mask = data.new(batch.mask.size(0)).fill_(1)
    return MaskedBatch(data, mask, ())

MaskedBatch.size_as_tensor = size_as_tensor
TENSOR_TYPE.size_as_tensor = size_as_tensor

def maxsize(batch, dim=None):
    return batch.data.size() if dim is None else batch.data.size(dim)

MaskedBatch.maxsize = maxsize
TENSOR_TYPE.maxsize = maxsize

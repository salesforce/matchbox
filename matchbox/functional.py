import torch
from torch.nn import functional as F
from torch.autograd import Variable

from . import MaskedBatch

def dropout(batch, p=0.5, training=False, inplace=False):
    if not isinstance(batch, MaskedBatch):
        return F.dropout(batch, p, training, inplace)
    data = F.dropout(batch.data, p, training, inplace)
    return MaskedBatch(data, batch.mask, batch.dims)

MaskedBatch.dropout = dropout
Variable.dropout = dropout

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
        data = data / data.sum(dim, keepdim=True) * batch.mask
        data[data.ne(data).detach()] = 0 # remove NaNs
        mask = batch.mask.narrow(dim, 0, 1)
        dims = dims[:dim - 1] + (False,) + dims[dim:]
    else:
        data = F.softmax(batch.data, dim)
        mask = batch.mask
    return MaskedBatch(data, mask, dims)

MaskedBatch.softmax = softmax
Variable.softmax = softmax

def matmul(batch1, batch2):
    if not isinstance(batch1, MaskedBatch) and not isinstance(batch2, MaskedBatch):
        return F.matmul(batch1, batch2)
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

def _elementwise_unary(fn):
    def inner(batch, **kwargs):
        if not isinstance(batch, MaskedBatch):
            return fn(batch, **kwargs)
        data = fn(batch.data, **kwargs)
        mask = batch.mask
        dims = batch.dims
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.relu = relu = _elementwise_unary(F.relu)
MaskedBatch.tanh = tanh = _elementwise_unary(F.tanh)
MaskedBatch.sigmoid = sigmoid = _elementwise_unary(F.sigmoid)

def _elementwise_binary(fn):
    def inner(batch1, batch2, **kwargs):
        if not isinstance(batch1, MaskedBatch) and not isinstance(batch2, MaskedBatch):
            return fn(batch1, batch2, **kwargs)
        if isinstance(batch2, MaskedBatch):
            mask = batch1.mask * batch2.mask
            data = fn(batch1.data, batch2.data, **kwargs)
            dims = tuple(b1 or b2 for b1, b2 in zip(batch1.dims, batch2.dims))
        else:
            mask = batch1.mask
            data = fn(batch1.data, batch2, **kwargs)
            dims = batch1.dims
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.__add__ = _elementwise_binary(torch.add)
MaskedBatch.__sub__ = _elementwise_binary(lambda a, b: a - b)
MaskedBatch.__mul__ = _elementwise_binary(torch.mul)
MaskedBatch.__truediv__ = _elementwise_binary(torch.div)

def _reduce(fn, zero_preserving=False):
    def inner(batch, dim=None, keepdim=False):
        if dim is None:
            if not zero_preserving and any(batch.dims):
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

def getitem(batch, index):
    if not isinstance(index, tuple) or index[0] != slice(None):
        raise ValueError("first index must be :")
    if None in index:
        raise NotImplementedError("cannot index with None")
    data = batch.data[index]
    mask = batch.mask[tuple(i if b else 0 if isinstance(i, int) else slice(None)
                       for i, b in zip(index, (True,) + batch.dims))]
    dims = tuple(b for i, b in zip(index[1:] + (slice(None),) * len(batch.dims),
                                   batch.dims)
                 if not isinstance(i, int)) # could be faster
    return MaskedBatch(data, mask, dims)

MaskedBatch.__getitem__ = getitem

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
Variable.unbind = unbind

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
                               for i in range(1, len(args)))
    mask = batch.mask.view(*mask_sizes) # TODO can this throw if data doesn't?
    dims = tuple(sizes[i] == -1 for i in range(1, len(args)))
    return MaskedBatch(data, mask, dims, batch)

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
Variable.transpose = transpose

def permute(batch, *permutation):
    data = batch.data.permute(*permutation)
    mask = batch.mask.permute(*permutation)
    dims = tuple(batch.dims[i - 1] for i in permutation[1:])
    return MaskedBatch(data, mask, dims)

MaskedBatch.permute = permute

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
Variable.split_dim = split_dim

def combine_dims(batch, dim1, dim2):
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

MaskedBatch.combine_dims = combine_dims
Variable.combine_dims = combine_dims

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
Variable.causal_mask = causal_mask

def maxsize(batch, dim=None):
    return batch.data.size() if dim is None else batch.data.size(dim)

MaskedBatch.maxsize = maxsize
Variable.maxsize = maxsize

def _synchronize(batch):
    if not isinstance(batch, MaskedBatch):
        return batch
    if any(batch.dims):
        raise ValueError("cannot synchronize batch with dynamic dimensions")
    mask = batch.mask + (1 - batch.mask)
    return MaskedBatch(batch.data, mask, batch.dims)

MaskedBatch._synchronize = _synchronize
Variable._synchronize = _synchronize

def _update(batch, new):
    if not isinstance(batch, MaskedBatch) and not isinstance(new, MaskedBatch):
        return new
    if isinstance(batch, MaskedBatch):
        data = torch.where(new.mask, new.data, batch.data)
    else:
        data = torch.where(new.mask, new.data, batch)
    mask = new.mask
    dims = new.dims
    return MaskedBatch(data, mask, dims)

MaskedBatch._update = _update
Variable._update = _update

# def _for(closure, iterator):
#     for i in iterator:
#         closure(i)

def _inject_arith(original, replacement):
    def inner(self, other):
        if isinstance(other, MaskedBatch):
            return replacement(self, other)
        return original(self, other)
    return inner

Variable.__add__ = _inject_arith(Variable.__add__, lambda a, b: b + a)
Variable.__sub__ = _inject_arith(Variable.__sub__, lambda a, b: -b + a)
Variable.__mul__ = _inject_arith(Variable.__mul__, lambda a, b: b * a)
# TODO
# Variable.__matmul__ = _inject_arith(Variable.__add__, lambda a, b:)
# Variable.__truediv__ = _inject_arith(Variable.__add__, lambda a, b:)

import sys
#torch.nn.functional = sys.modules[__name__] # monkeys in the bamboo tree
import torch.nn.modules.sparse
torch.nn.modules.sparse.F = sys.modules[__name__]
import torch.nn.modules.linear
torch.nn.modules.linear.F = sys.modules[__name__]
import torch.nn.modules.dropout
torch.nn.modules.dropout.F = sys.modules[__name__]

import torch.nn._functions.rnn
torch.nn._functions.rnn.F = sys.modules[__name__]

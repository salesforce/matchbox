import torch
from torch.nn import functional as F

from . import MaskedBatch

def dropout(input, p=0.5, training=False, inplace=False):
    data = F.dropout(input.data, p, training, inplace)
    return MaskedBatch(data, input.mask, input.dims)

def linear(input, weight, bias=None):
    """
    Applies a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    if input.dims[-1]:
        raise ValueError("cannot contract static and dynamic dimensions")
    data = F.linear(input.data, weight, bias)
    return MaskedBatch(data, input.mask, input.dims)

def embedding(input, weight, padding_idx=None, max_norm=None, norm_type=2,
              scale_grad_by_freq=False, sparse=False):
    r"""A simple lookup table that looks up embeddings in a fixed dictionary and size.
    This module is often used to retrieve word embeddings using indices.
    The input to the module is a list of indices, and the embedding matrix,
    and the output is the corresponding word embeddings.
    Args:
        input: tensor, containing indices into the embedding matrix
        weight:
            Number of rows should correspond to the maximum possible index + 1,
            number of columns is the embedding size
        max_norm (float, optional): If given, will renormalize the embeddings to always have a norm lesser than this
        norm_type (float, optional): The p of the p-norm to compute for the max_norm option
        scale_grad_by_freq (boolean, optional): if given, this will scale gradients by the frequency of
                                                the words in the mini-batch.
        sparse (boolean, optional): if ``True``, gradient w.r.t. weight matrix will be a sparse tensor. See Notes for
                                    more details regarding sparse gradients.
    Shape:
        - Input: LongTensor `(N, W)`, N = mini-batch, W = number of indices to extract per mini-batch
        - Embedding_matrix: FloatTensor `(V, embedding_dim)`, V = maximum index + 1, embedding_dim = embedding size
        - Output: `(N, W, embedding_dim)`
    Notes:
        It is advised to only use `sparse=True` if `embedding_matrix` is a leaf Variable,
        since some autograd functions may not propagate sparse gradients correctly.
        Additionally, keep in mind that only a limited number of optimizers support
        sparse gradients: currently it's `optim.SGD` (`cuda` and `cpu`), and `optim.Adagrad` (`cpu`)
    Examples::
        >>> # a batch of 2 samples of 4 indices each
        >>> input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]))
        >>> # an embedding matrix containing 10 tensors of size 3
        >>> embedding_matrix = Variable(torch.rand(10, 3))
        >>> F.embedding(input, embedding_matrix)
        Variable containing:
        (0 ,.,.) =
         -1.0822  1.2522  0.2434
          0.8393 -0.6062 -0.3348
          0.6597  0.0350  0.0837
          0.5521  0.9447  0.0498
        (1 ,.,.) =
          0.6597  0.0350  0.0837
         -0.1527  0.0877  0.4260
          0.8393 -0.6062 -0.3348
         -0.8738 -0.9054  0.4281
        [torch.FloatTensor of size 2x4x3]
        >>> # example with padding_idx
        >>> weights = torch.rand(10, 3)
        >>> weights[0, :].zero_()
        >>> embedding_matrix = Variable(weights)
        >>> input = Variable(torch.LongTensor([[0,2,0,5]]))
        >>> F.embedding(input, embedding_matrix, padding_idx=0)
        Variable containing:
        (0 ,.,.) =
          0.0000  0.0000  0.0000
          0.3452  0.4937 -0.9361
          0.0000  0.0000  0.0000
          0.0706 -2.1962 -0.6276
        [torch.FloatTensor of size 1x4x3]
    """
    #data = input.data - input.mask
    data = input.data
    if torch.__version__ < '0.4':
        data = F.embedding(
            data, weight, max_norm, norm_type, scale_grad_by_freq, sparse)
    else:
        data = F.embedding(
            data, weight, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)
    mask = input.mask.unsqueeze(-1).float()
    data = data * mask
    dims = input.dims + (False,)
    return MaskedBatch(data, mask, dims)

def softmax(input, dim=-1):
    r"""Applies a softmax function.
    Softmax is defined as:
    :math:`softmax(x) = \frac{exp(x_i)}{\sum_j exp(x_j)}`
    It is applied to all slices along dim, and will rescale them so that the elements
    lie in the range `(0, 1)` and sum to 1.
    See :class:`~torch.nn.Softmax` for more details.
    Arguments:
        input (Variable): input
        dim (int): A dimension along which softmax will be computed.
    .. note::
        This function doesn't work directly with NLLLoss,
        which expects the Log to be computed between the Softmax and itself.
        Use log_softmax instead (it's faster and has better numerical properties).
    """
    if dim == 0:
        raise ValueError("cannot softmax over batch dimension")
    elif dim < 0:
        dim = input.data.dim() + dim
    dims = input.dims
    if dims[dim - 1]:
        data = F.softmax(input.data * input.mask, dim) * input.mask
        data = data / data.sum(dim, keepdim=True) * input.mask
        data[data.ne(data).detach()] = 0 # remove NaNs
        mask = input.mask.narrow(dim, 0, 1)
        dims = dims[:dim - 1] + (False,) + dims[dim:]
    else:
        data = F.softmax(input.data, dim)
        mask = input.mask
    return MaskedBatch(data, mask, dims)

def matmul(batch1, batch2, out=None):
    r"""Matrix product of two tensors.
    Behavior may differ from the below docstring for now...

    The behavior depends on the dimensionality of the tensors as follows:
    - If both tensors are 1-dimensional, the dot product (scalar) is returned.
    - If both arguments are 2-dimensional, the matrix-matrix product is returned.
    - If the first argument is 1-dimensional and the second argument is 2-dimensional,
      a 1 is prepended to its dimension for the purpose of the matrix multiply.
      After the matrix multiply, the prepended dimension is removed.
    - If the first argument is 2-dimensional and the second argument is 1-dimensional,
      the matrix-vector product is returned.
    - If both arguments are at least 1-dimensional and at least one argument is
      N-dimensional (where N > 2), then a batched matrix multiply is returned.  If the first
      argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the
      batched matrix multiply and removed after.  If the second argument is 1-dimensional, a
      1 is appended to its dimension for the purpose of the batched matrix multiple and removed after.
      The non-matrix (i.e. batch) dimensions are :ref:`broadcasted <broadcasting-semantics>` (and thus
      must be broadcastable).  For example, if :attr:`tensor1` is a
      :math:`(j \times 1 \times n \times m)` tensor and :attr:`tensor2` is a :math:`(k \times m \times p)`
      tensor, :attr:`out` will be an :math:`(j \times k \times n \times p)` tensor.
    .. note::
        The 1-dimensional dot product version of this function does not support an :attr:`out` parameter.
    Arguments:
        batch1 (MaskedBatch): the first tensor to be multiplied
        batch2 (MaskedBatch): the second tensor to be multiplied
    """
    if isinstance(batch1, MaskedBatch) and isinstance(batch2, MaskedBatch):
        dim_batch1 = len(batch1.dims)
        dim_batch2 = len(batch2.dims)
        if out is not None:
            raise NotImplementedError("matmul with out argument not implemented")
        if dim_batch1 == 1 and dim_batch2 == 1:
            if (batch1.dims[0] or batch2.dims[0]) and not batch1.mask.eq(batch2.mask).all():
                raise ValueError("cannot contract non-matching dimensions")
            data = batch1.data.unsqueeze(-2) @ batch2.data.unsqueeze(-1)
            mask = batch1.mask[:, :1]
            dims = ()
        if dim_batch1 == 2 and dim_batch2 == 1:
            if (batch1.dims[1] or batch2.dims[0]) and not batch1.mask[:, 0].eq(batch2.mask).all():
                raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :, :1] @ batch2.mask[:, :1]
            data = batch1.data @ batch2.data
            dims = batch1.dims[:1]
        elif dim_batch1 == 1 and dim_batch2 == 2:
            if (batch1.dims[0] or batch2.dims[0]) and not batch1.mask.eq(batch2.mask[:, :, 0]).all():
                raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :1].unsqueeze(-2) @ batch2.mask[:, :1, :]
            data = batch1.data.unsqueeze(-2) @ batch2.data
            dims = batch2.dims[1:]
        elif dim_batch1 == 2 and dim_batch2 == 2:
            if (batch1.dims[1] or batch2.dims[0]) and not batch1.mask[:, 0].eq(batch2.mask[:, :, 0]).all():
                raise ValueError("cannot contract non-matching dimensions")
            mask = batch1.mask[:, :, :1] @ batch2.mask[:, :1, :]
            data = batch1.data @ batch2.data
            dims = batch1.dims[:1] + batch2.dims[1:]
        else:
            raise NotImplementedError("matmul not implemented with batches of 3+D tensors")
    else:
        raise NotImplementedError("matmul not implemented between MaskedBatch and tensor")
    return MaskedBatch(data, mask, dims)

MaskedBatch.__matmul__ = matmul

def _elementwise_unary(fn, zero_preserving=False):
    def inner(batch, **kwargs):
        data = fn(batch.data, **kwargs)
        mask = batch.mask
        dims = batch.dims
        if not zero_preserving:
            data *= mask
        return MaskedBatch(data, mask, dims)
    return inner

def _elementwise_binary(fn, identity):
    def inner(batch1, batch2, **kwargs):
        if isinstance(batch2, MaskedBatch):
            if identity is None:
                raise ValueError("binary elementwise operations require an identity")
            data1, imask1 = batch1.data, 1 - batch1.mask
            data2, imask2 = batch2.data, 1 - batch2.mask
            if identity != 0:
                data1 = data1 + imask1 * identity
                data2 = data2 + imask2 * identity
            data = fn(data1, data2, **kwargs)
            if batch1.dims != batch2.dims:
                raise NotImplementedError("binary elementwise operations currently "
                                          "require matching static/dynamic dims")
            mask = 1 - imask1 * imask2
            dims = batch1.dims
        else:
            data = fn(batch1.data, batch2, **kwargs)
            mask = batch1.mask
            dims = batch1.dims
            if identity != 0:
                data *= mask
        return MaskedBatch(data, mask, dims)
    return inner

MaskedBatch.relu = relu = _elementwise_unary(F.relu, zero_preserving=True)
MaskedBatch.tanh = tanh = _elementwise_unary(F.tanh, zero_preserving=True)
MaskedBatch.sigmoid = sigmoid = _elementwise_unary(F.sigmoid)
MaskedBatch.__add__ = _elementwise_binary(torch.add, identity=0)
MaskedBatch.__sub__ = _elementwise_binary(lambda a, b: a - b, identity=0)
MaskedBatch.__mul__ = _elementwise_binary(torch.mul, identity=1)
MaskedBatch.__div__ = _elementwise_binary(torch.div, identity=1)

def _reduce(fn, zero_preserving=False):
    def inner(batch, dim=None, keepdim=False):
        if dim is None:
            if not zero_preserving and any(batch.dims):
                raise NotImplementedError(
                    "cannot reduce to scalar with non-zero-preserving kernel "
                    "if dynamic dims present")
            mask = batch.mask[(slice(None), *(0 for d in input.dims))]
            dims = ()
        else:
            if dim < 0:
                dim = input.data.dim() + dim
            if not zero_preserving and batch.dims[dim - 1]:
                raise NotImplementedError(
                    "cannot reduce over dynamic dim with non-zero-preserving kernel")
            if keepdim:
                mask = batch.mask[(slice(0, 1) if i == dim else slice(None))]
                dims = tuple(
                    False if i == dim - 1 else d for i, d in enumerate(input.dims))
            else:
                mask = batch.mask[(0 if i == dim else slice(None))]
                dims = tuple(d for i, d in enumerate(input.dims) if i != dim - 1)
        data = fn(batch.data, dim=dim, keepdim=keepdim)
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
    dims = tuple(b for i, b in zip(index[1:] + (slice(None),) * len(batch.dims), batch.dims)
                 if not isinstance(i, int)) # could be faster
    return MaskedBatch(data, mask, dims)

MaskedBatch.__getitem__ = getitem

def unbind(batch, dim):
    if dim == 0:
        raise ValueError("cannot unbind over batch dimension")
    dims = tuple(b for d, b in enumerate(batch.dims) if d != dim - 1)
    if batch.dims[dim - 1]:
        for data, mask in zip(torch.unbind(batch.data, dim), torch.unbind(batch.mask, dim)):
            yield MaskedBatch(data, mask, dims)
    else:
        mask = batch.mask.squeeze(dim)
        for data in torch.unbind(batch.data, dim):
            yield MaskedBatch(data, mask, dims)

#MaskedBatch.unbind = unbind

def size(batch, dim=None):
    if dim is None:
        return
    if dim > 0 and batch.dims[dim - 1]:
        raise NotImplementedError("cannot get size of dynamic dim")
    else:
        return batch.data.size(dim)

MaskedBatch.size = size

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
    return MaskedBatch(data, mask, dims)

MaskedBatch.view = view

# def _for(closure, iterator):
#     for i in iterator:
#         closure(i)

import sys
torch.nn.functional = sys.modules[__name__] # monkeys in the bamboo tree
import torch.nn.modules.sparse
torch.nn.modules.sparse.F = sys.modules[__name__]
import torch.nn.modules.linear
torch.nn.modules.linear.F = sys.modules[__name__]
import torch.nn.modules.dropout
torch.nn.modules.dropout.F = sys.modules[__name__]

import torch.nn._functions.rnn
torch.nn._functions.rnn.F = sys.modules[__name__]

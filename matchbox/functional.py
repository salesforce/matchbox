import torch
from torch.nn import functional as F

from . import MaskedBatch

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

def relu(input, inplace=False):
    """relu(input, threshold, value, inplace=False) -> Variable
    Applies the rectified linear unit function element-wise. See
    :class:`~torch.nn.ReLU` for more details.
    """
    data = F.relu(input.data, inplace)
    return MaskedBatch(data, input.mask, input.dims)

def softmax(input, dim=None):
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
        data[data.ne(data)] = 0 # remove NaNs
        mask = input.mask.narrow(dim, 0, 1)
        dims = dims[:dim - 1] + (False,) + dims[dim:]
    else:
        data = F.softmax(input.data, dim)
        mask = input.mask
    return MaskedBatch(data, mask, dims)

def matmul(tensor1, tensor2, out=None):
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
        tensor1 (Tensor): the first tensor to be multiplied
        tensor2 (Tensor): the second tensor to be multiplied
        out (Tensor, optional): the output tensor
    """
    dim_tensor1 = len(tensor1.dims)
    dim_tensor2 = len(tensor2.dims)
    if out is not None:
        raise NotImplementedError("matmul with out argument not implemented")
    if dim_tensor1 == 1 and dim_tensor2 == 1:
        if tensor1.mask != tensor2.mask:
            raise ValueError("cannot contract non-matching dimensions")
        data = tensor1.data.unsqueeze(-2) @ tensor2.data.unsqueeze(-1)
        mask = tensor1.mask[:, 0]
        dims = ()
    if dim_tensor1 == 2 and dim_tensor2 == 1:
        if tensor1.mask[:, 0] != tensor2.mask:
            raise ValueError("cannot contract non-matching dimensions")
        data = tensor1.data @ tensor2.data
        mask = tensor1.mask[:, :, 0]
        dims = tensor1.dims[:1]
    elif dim_tensor1 == 1 and dim_tensor2 == 2:
        if tensor1.mask != tensor2.mask[:, :, 0]:
            raise ValueError("cannot contract non-matching dimensions")
        data = tensor1.data.unsqueeze(-2) @ tensor2.data
        mask = tensor2.mask[:, 0, :]
        dims = tensor2.dims[1:]
    elif dim_tensor1 == 2 and dim_tensor2 == 2:
        if tensor1.mask[:, 0] != tensor2.mask[:, :, 0]:
            raise ValueError("cannot contract non-matching dimensions")
        data = tensor1.data @ tensor2.data
        mask = tensor1.mask[:, :, 0]
        dims = tensor1.dims[:1] + tensor2.dims[1:]
    else:
        raise NotImplementedError("matmul not implemented with batches of 3+D tensors")
    return MaskedBatch(data, mask, dims)

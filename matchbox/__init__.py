import torch

TENSOR_TYPE = (torch.autograd.Variable if torch.__version__ < '0.4'
               else torch.Tensor)

class MaskedBatch(object):

    def __init__(self, data, mask, dims):
        if data.dim() != mask.dim() or mask.dim() != len(dims) + 1:
            raise ValueError("malformed MaskedBatch {} with:\n data: {}\n mask: {}".format(
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
        sizes = [max(x.maxsize(d + 1) for x in examples)
                 for d in range(len(dims))]
        data = examples[0].new(bs, *sizes).zero_()
        mask_sizes = [s if dims[d] else 1 for d, s in enumerate(sizes)]
        mask = examples[0].new(bs, *mask_sizes).zero_()
        mask.requires_grad = False
        for i, x in enumerate(examples):
            inds = [slice(0, x.maxsize(d + 1)) if b else slice(None)
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

    def new_zeros(self, *sizes):
        return self.data.new_zeros(*sizes)
        # mask = batch.mask + (1 - batch.mask)
        # return MaskedBatch()

    def __bool__(self):
        if self.data.nelement() > 1:
            raise ValueError("bool value of MaskedBatch with more than one "
                             "value is ambiguous; use .any() or .all() or wrap "
                             "code containing control flow in @batch.")
        return bool(self.data)

from . import functional
from . import data
from . import recompile
from . import macro
from .macro import batch

if torch.__version__ < '0.4':
    def _var_method(method_name):
        def inner(self, *args, **kwargs):
            t = getattr(self.data, method_name)(*args, **kwargs)
            return torch.autograd.Variable(t)
        return inner
    torch.autograd.Variable.new = _var_method('new')

    def _where(cond, x, y):
        cond = cond.type_as(x)
        return x * cond + y * (1 - cond)
    torch.where = _where

    _old_arange = torch.arange
    def _new_arange(*args, out=None):
        if isinstance(out, torch.autograd.Variable):
            torch.arange(*args, out=out.data)
            return out
        return _old_arange(*args, out=out)
    torch.arange = _new_arange

    def embed_forward(self, input):
        return functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
    torch.nn.Embedding.forward = embed_forward

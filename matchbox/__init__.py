import torch

class MaskedBatch(object):

    def __init__(self, data, mask, dims):
        if data.dim() != mask.dim() or mask.dim() != len(dims) + 1:
            raise ValueError("malformed MaskedBatch {} with:\n data: {}\n mask: {}".format(
                repr(dims), repr(data), repr(mask)))
        if isinstance(mask, torch.autograd.Variable) and mask.requires_grad:
            raise ValueError("mask cannot require grad")
        self.data = data
        self.mask = mask
        self.dims = dims

    @classmethod
    def fromlist(cls, examples, dims):
        # do some validation
        bs = len(examples)
        sizes = [max(x.size(d + 1) for x in examples) for d in range(len(dims))]
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
        for i in range(data.size(0)):
            inds = tuple(slice(0, mask[i].sum(d, keepdim=True)[tuple(0 for _ in dims)])
                         if b else slice(None) for d, b in enumerate(dims))
            yield data[(slice(i, i + 1), *inds)]

    def __repr__(self):
        return "MaskedBatch {} with:\n data: {}\n mask: {}".format(
            repr(self.dims), repr(self.data), repr(self.mask))

    def cuda(self):
        data = self.data.cuda()
        mask = self.mask.cuda()
        return self.__class__(data, mask, self.dims)

    @property
    def is_cuda(self):
        return self.data.is_cuda

    def get_device(self):
        return self.data.get_device()

    def transpose(self, dim1, dim2):
        data = self.data.transpose(dim1, dim2)
        mask = self.mask.transpose(dim1, dim2)
        dims = list(self.dims)
        dims[dim1 - 1], dims[dim2 - 1] = dims[dim2 - 1], dims[dim1 - 1]
        dims = tuple(dims)
        return self.__class__(data, mask, dims)

if torch.__version__ < '0.4':
    def var_new(self, *args, **kwargs):
        n = self.data.new(*args, **kwargs)
        return torch.autograd.Variable(n)
    torch.autograd.Variable.new = var_new

    def embed_forward(self, input):
        return torch.nn.functional.embedding(
            input, self.weight, self.padding_idx, self.max_norm,
            self.norm_type, self.scale_grad_by_freq, self.sparse)
    torch.nn.Embedding.forward = embed_forward

from . import functional
from . import data

import torch

class AbstractBatch(object):
    pass

class VectorBatch(AbstractBatch):
    pass

class MaskedBatch(AbstractBatch):

    def __init__(self, data, mask, dims):
        if data.dim() != mask.dim() or mask.dim() != len(dims) + 1:
            raise ValueError("malformed batch object")
        self.data = data
        self.mask = mask
        self.dims = dims

    @classmethod
    def fromlist(cls, examples, dims):
        # do some validation
        bs = len(examples)
        sizes = (max(x.size(d + 1) for x in examples) for d in range(len(dims)))
        data = examples[0].new(bs, *sizes).zero_()
        mask_sizes = (s if dims[d] else 1 for d, s in enumerate(sizes))
        mask = examples[0].new(bs, *mask_sizes).zero_()
        for i, x in enumerate(examples):
            inds = (slice(0, x.size(d + 1)) if b else slice(None)
                    for d, b in enumerate(dims))
            data.__setitem__(i, *inds, x)
            mask.__setitem__(i, *inds, 1)
        return cls(data, mask, dims)

    def __repr__(self):
        return "MaskedBatch with:\n data: {}\n mask: {}".format(
            repr(self.data), repr(self.mask))

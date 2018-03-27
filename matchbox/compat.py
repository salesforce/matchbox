import torch

if torch.__version__ < '0.4':
    MAYBE_VARIABLE = TENSOR_TYPE = torch.autograd.Variable
else:
    def identity(x): return x
    MAYBE_VARIABLE = identity
    TENSOR_TYPE = torch.Tensor

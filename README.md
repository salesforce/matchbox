# Matchbox

Matchbox enables deep learning researchers to write PyTorch code at the level
of individual examples, then run it efficiently on minibatches. It does this
using three components:
- A `MaskedBatch` type, together with overloaded implementations of PyTorch
methods and neural network layers, keeps track of padding and masking for
variable-size data automatically. Use `dir(matchbox.MaskedBatch)` to see a list
of supported methods.
- A `@batch` decorator rewrites some Python control flow into a
[SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)-like
form that includes execution masking and synchronization primitives.
- Convenience methods like `batch_ones`, `split_dim`, and `causal_mask` support
common use cases in dynamic neural network code in a way that benefits from
the more semantically meaningful shape information available with the
`MaskedBatch` type. These are implemented both for batch and tensor objects,
because all code written for Matchbox also works with plain `Tensor`s at batch
size one.

There is also a plugin for [torchtext](https://github.com/pytorch/text) and a
wrapper for testing that Matchbox results are numerically equivalent to a loop
over unbatched examples. See the `examples` and `test` directories for details.

## Installation and requirements
Matchbox is in early-release alpha. Use `python setup.py install` to install.
Please file or upvote issues to request new operation implementations, or feel
free to post one as a pull request. If Matchbox throws a `NotImplementedError`,
that means that a particular feature of an operation could be supported but
isn't yet.

Matchbox is developed on Python 3.6 and PyTorch 0.4. It contains compatibility
code that is intended to support PyTorch 0.3, but not all features will work.
Matchbox also requires `gast`, `astor`, and `six`. Python 2 support is not an
immediate priority but we would welcome a PR.

## Getting started
The first step to using Matchbox is to replace your import of
`torch.nn.functional` with `matchbox.functional`:
```python
import matchbox
import matchbox.functional as F
# now calls like `F.softmax` refer to Matchbox's implementations
```
This import also replaces methods on PyTorch `Tensor`s with Matchbox versions
and injects `matchbox.functional` functions into `torch.nn` modules.

Now you can write model code that applies to individual examples. If your code
uses control flow, add the `@matchbox.batch` decorator to that function or
class (unfortunately, this doesn't yet work in the interactive interpreter
or in Jupyter notebooks):
```python
from torch import nn
class RNN(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.cell = nn.RNNCell(size, size)
    @matchbox.batch
    def forward(self, x):
        h = x.new_zeros(x.size(0), x.size(-1))
        for xt in x.unbind(1):
            h = self.cell(xt, h)
        return h
```

You can create input data to pass to this model in three ways. First, you can
pass them ordinary PyTorch `Tensor`s with batch size one. You can also pass
`MaskedBatch` objects created manually, from lists of `Tensor`s with batch
size one (note that `torch.rand` should be wrapped in `Variable` on PyTorch
0.3):
```python
import torch
from matchbox import MaskedBatch
from random import randint
b, t, c = 32, 10, 128
model = RNN(c)
x_unbatched = torch.rand(1, randint(1, t), c) # a single random example
x_manual_batch = MaskedBatch.fromlist(
    [torch.rand(1, randint(1, t), c) for i in range(b)], # list of examples
    (True, False)) # dimension 1 is dynamic and dimension 2 is static
h = model(x_unbatched)
h = model(x_manual_batch)
```
And we provide a `torchtext` Field class that produces `MaskedBatch` objects
when a dataset is iterated:
```python
from matchbox.data import MaskedBatchField
TEXT = MaskedBatchField(batch_first=True)
train, dev, test = datasets.IWSLT.splits(('.de', '.en'), (TEXT, TEXT))
TEXT.build_vocab(train, max_size=50000)
train_iter = data.BucketIterator(train, batch_size=32, device=-1)
for x_torchtext_batch in train_iter:
    h = model(x_torchtext_batch)
    # more training loop code
```
## Credit
Matchbox is developed by James Bradbury at Salesforce Research.
It also contains Python source-wrangling code modified from Patrick Maupin
and Berker Peksag's
[AST observe-rewrite](https://github.com/berkerpeksag/astor) as well as
Google Brain's [Tangent](https://github.com/google/tangent), a source-to-source
automatic differentiation package developed by Alex Wiltschko, Bart van
Merrienboer and Dan Moldovan. The modified Tangent code is licensed under
Apache 2 while the rest of the codebase is licensed under three-clause BSD;
see `LICENSE.BSD-3.txt` and `LICENSE.Apache-2.txt`.

## Limitations
Matchbox only works on code that uses native PyTorch operators. In particular,
everything that could vary between examples in a batch needs to be a `Tensor`
in order for code written for individual examples to work with Matchbox. Support
for scalar tensors is significantly better in PyTorch 0.4. NumPy ops also need
to be replaced with their native PyTorch equivalents.

Control flow support is limited. While some of these limitations will be lifted
(e.g., support for `continue` within `while` is straightforward to add) some
constructs are conceptually harder for Matchbox to support (e.g., `return` from
within a `for`).

There’s also a long tail of less-common operations that haven’t been
implemented (plus bigger gaps, like convolutions). We will be continuously
adding support for additional ops but also welcome pull requests.

## Implementation details (batch semantics)
`MaskedBatch` objects behave like PyTorch `Tensor`s, but represent a
collection ("batch") of `Tensor`s that may be of different sizes in some
of their dimensions.
Most of the time, `MaskedBatch` objects adhere to Matchbox's "standard"
semantics, but control flow constructions require a different "SIMT"
semantics.
### Standard
The `dims` attribute is a `tuple` with a `bool` for each non-batch dimension,
representing whether that dimension is static (`False`) or dynamic (`True`).

The `data` attribute is a `Tensor` whose size is the batch size in the batch
dimension, the size of all examples in static dimensions, and at least as large
as the largest example in the batch in dynamic dimensions.

The `mask` attribute is a `Tensor` whose size is the batch size in the batch
dimension, one in static dimensions, and at least as large as the largest
example in the batch in dynamic dimensions. Each entry in the mask corresponds
to one or more entries in the data array (singleton, i.e., static, dimensions
are broadcasted), with a one in the mask denoting that the corresponding data
entries represent valid, meaningful data and a zero denoting that they do not.

Data values corresponding to zeros in the mask are not required to be zero,
and operations should propagate masked data if doing so would not affect
non-masked parts of the output. Operations for which this is not the case
should first multiply their input data by the corresponding masks.
### SIMT
A one in the mask denotes that the corresponding data entries represent
currently active data. A zero denotes that the corresponding data entries
represent "dormant" data, which may be valid at a previous step of a loop
(e.g., at a previous index along an external dimension that is being iterated
over) or in another branch of a conditional. Currently, no dimensions in a
SIMT batch may be dynamic, but support for this case will be added.

## Future work
In addition to adding `MaskedBatch` support for more operations, we also plan
a separate `PackedBatch` type that can pack its data tensor along its batch
dimension and one dynamic dimension and store a separate tensor of offsets.
This type will be natively compatible with cuDNN RNNs and saves memory relative
to `MaskedBatch`, but will be slower for some operations.

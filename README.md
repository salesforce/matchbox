# matchbox

Matchbox enables deep learning researchers to write PyTorch code at the level
of individual examples, then run it efficiently on minibatches. It does this
using three components:
- A `MaskedBatch` type, together with overloaded implementations of PyTorch
methods and neural network layers, keeps track of padding and masking for
variable-size data automatically. Use `dir(matchbox.MaskedBatch)` to see a list
of supported methods.
- A `batch` decorator rewrites some Python control flow into a
[SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads)-like
form that includes execution masking and synchronization primitives.
- Convenience methods like `batch_ones`, `split_dim`, and `causal_mask` support
common use cases in dynamic neural network code in a way that benefits from
the more semantically meaningful shape information available with the
`MaskedBatch` type. These are implemented both for batch and tensor objects,
because all code written for Matchbox also works with plain Tensors at batch
size one.

There is also a plugin for [torchtext](https://github.com/pytorch/text) and a
wrapper for testing that Matchbox results are numerically equivalent to a loop
over unbatched examples. See the `examples` and `test` directories for details.

Matchbox is in early-release beta. Please file issues to request new operation
implementations, or feel free to post one as a pull request.

Matchbox is developed on PyTorch master (i.e., what will soon be released
as version 0.4). It contains compatibility code that is intended to support
PyTorch 0.3, but not all features will work. Matchbox also requires `gast` and
`astor` and contains additional Python source-wrangling code from
[Tangent](https://github.com/google/tangent), used under an Apache 2 license.

## Implementation Details (batch semantics)
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
Data values corresponding to zeros in the mask are required to be zero, as some
operations take advantage of this fact for efficiency purposes.
### SIMT
A one in the mask denotes that the corresponding data entries represent
currently active data. A zero denotes that the corresponding data entries
represent "dormant" data, which may be valid at a "previous" index along
an external dimension that is being iterated over or in another branch of
a conditional. Currently, no dimensions in a SIMT batch may be dynamic, but
support for this case will be added.

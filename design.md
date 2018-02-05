#Semantics
`MaskedBatch` objects behave like PyTorch `Variable`s, but represent a
collection ("batch") of `Variable`s that may be of different sizes in some
of their dimensions.
Most `MaskedBatch` objects adhere to Matchbox's "standard" semantics, but some
loop constructions require a different "accumulating" semantics.
##Standard
The `dims` attribute is a `tuple` with a `bool` for each non-batch dimension,
representing whether that dimension is static (`False`) or dynamic (`True`).
The `data` attribute is a `Variable` whose size is the batch size in the batch
dimension, the size of all examples in static dimensions, and at least as large
as the largest example in the batch in dynamic dimensions.
The `mask` attribute is a `Variable` whose size is the batch size in the batch
dimension, one in static dimensions, and at least as large as the largest
example in the batch in dynamic dimensions. Each entry in the mask corresponds
to one or more entries in the data array (singleton, i.e., static, dimensions
are broadcasted), with a one in the mask denoting that the corresponding data
entries represent valid, meaningful data and a zero denoting that they do not.
Data values corresponding to zeros in the mask are required to be zero, as some
operations take advantage of this fact for efficiency purposes.
##Accumulating
A one in the mask denotes that the corresponding data entries represent
currently active data. A zero denotes that the corresponding data entries
represent "dormant" data, which may be valid at a "previous" index along
an external dimension that is being iterated over. No dimensions in an
accumulating batch may be dynamic.

Some options:
accumulating could be a flag
accumulating could be a separate type (definitely would in Julia)
accumulating could be a MaskedBatch with `dims[0]`, if dims also included the batch.
accumulating could be a MaskedBatch with `dims is None` or another sentinel
  I'm liking this; would be a good default promotion from Variable
accumulating could be every MaskedBatch with `not any(dims)`
  but a slice of a batch along the only dynamic dimension can't be accumulating,
  because masked-out data is actually invalid, not dormant
    unless ALL batches had "smeared" representations?
    this sounds like an OK idea, but actually won't work
    the reductions need identity elements, which means they need a privileged side

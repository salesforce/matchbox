import torch
from torch.autograd import Variable
from torchtext import data

from . import MaskedBatch

class MaskedBatchField(data.Field):

    def process(self, batch, device, train):
        """Process a list of examples to create a matchbox.MaskedBatch.

        Args:
            batch (list(object)): A list of object from a batch of examples.
        Returns:
            matchbox.MatchBox: Processed object given the input and custom
                postprocessing Pipeline.
        """
        batch = list(batch)
        if self.sequential:
            if self.fix_length is not None:
                raise ValueError("cannot use fix_length with Matchbox")
            batch = [([] if self.init_token is None else [self.init_token]) +
                     list(x) +
                     ([] if self.eos_token is None else [self.eos_token])
                     for x in batch]

        if self.use_vocab:
            if self.sequential:
                batch = [[self.vocab.stoi[x] for x in ex] for ex in batch]
            else:
                batch = [self.vocab.stoi[x] for x in batch]

            if self.postprocessing is not None:
                batch = self.postprocessing(batch, self.vocab, train)
        else:
            if self.tensor_type not in self.tensor_types:
                raise ValueError(
                    "Specified Field tensor_type {} can not be used with "
                    "use_vocab=False because we do not know how to numericalize it. "
                    "Please raise an issue at "
                    "https://github.com/pytorch/text/issues".format(self.tensor_type))
            numericalization_func = self.tensor_types[self.tensor_type]
            batch = [numericalization_func(x) if isinstance(x, six.string_types)
                   else x for x in batch]
            if self.postprocessing is not None:
                batch = self.postprocessing(batch, None, train)

        batch = [Variable(self.tensor_type(x).unsqueeze(0), volatile=not train) for x in batch]
        if self.sequential and not self.batch_first:
            raise ValueError("Matchbox requires batch_first for sequential Fields")
        dims = (True,) if self.sequential else ()
        batch = MaskedBatch.fromlist(batch, dims)
        if device != -1:
            batch = batch.cuda()
        return batch

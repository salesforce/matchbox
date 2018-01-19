import torch
from torch import nn
from torchtext import data, datasets
from matchbox import functional as F
from matchbox.data import MaskedBatchField

TEXT = MaskedBatchField(batch_first=True)
train, dev, test = datasets.IWSLT.splits(('.de', '.en'), (TEXT, TEXT))
TEXT.build_vocab(train)
train_iter = data.BucketIterator(train, batch_size=8)

batch = next(iter(train_iter))
print(batch.src)
print(batch.trg)

nn.functional = F
embed = nn.Embedding(len(TEXT.vocab), 10).cuda()
src = embed(batch.src)
trg = embed(batch.trg)
print(src)
print(trg)

attn = src @ trg.transpose(1, 2)
print(attn)

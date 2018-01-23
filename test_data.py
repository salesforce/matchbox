import torch
from torch import nn
from torchtext import data, datasets
import matchbox
from matchbox import functional as F
from matchbox.data import MaskedBatchField

TEXT = MaskedBatchField(batch_first=True)
train, dev, test = datasets.IWSLT.splits(('.de', '.en'), (TEXT, TEXT))
TEXT.build_vocab(train)
train_iter = data.BucketIterator(train, batch_size=8, device=-1)

batch = next(iter(train_iter))
print(batch.src)
print(batch.trg)

embed = nn.Embedding(len(TEXT.vocab), 10)
src = embed(batch.src)
trg = embed(batch.trg)
print(src)
print(trg)

alphas = src @ trg.transpose(1, 2)
print(alphas)

attns = F.softmax(alphas, -1)
print(attns)

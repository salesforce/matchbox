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

class LayerNorm(nn.Module):

    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualBlock(nn.Module):

    def __init__(self, layer, d_model, drop_ratio, pos=0):
        super().__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)
        self.pos = pos

    def forward(self, *x):
        return self.layernorm(x[self.pos] + self.dropout(self.layer(*x)))

class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal, diag=False, window=-1, noisy=False):
        super().__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
        self.diag = diag
        self.window = window
        self.noisy = noisy

    def forward(self, query, key, value=None)):
        dot_products = query @ key.transpose(1, 2)   # batch x trg_len x trg_len

        if query.dim() == 3 and self.causal and (query.size(1) == key.size(1)):
            tri = key.data.new(key.size(1), key.size(1)).fill_(1).triu(1) * INF
            dot_products.data.sub_(tri.unsqueeze(0)) # TODO

        if self.diag:
            inds = torch.arange(0, key.size(1)).long().view(1, 1, -1)
            if key.is_cuda:
                inds = inds.cuda(key.get_device())
            dot_products.data.scatter_(1, inds.expand(dot_products.size(0), 1, inds.size(-1)), -INF)
            # eye = key.data.new(key.size(1), key.size(1)).fill_(1).eye() * INF
            # dot_products.data.sub_(eye.unsqueeze(0))

        if value is None:
            return dot_products

        return self.dropout(F.softmax(dot_products / self.scale, -1)) @ value

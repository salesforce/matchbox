import torch
from torch import nn
from torchtext import data, datasets
import matchbox
from matchbox import functional as F
from matchbox.data import MaskedBatchField

import math

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

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))

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

    def forward(self, query, key, value=None):
        dot_products = query @ key.transpose(1, 2)   # batch x trg_len x trg_len

        if query.dim() == 3 and self.causal:# and (query.size(1) == key.size(1)):
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

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio,
                causal=False, diag=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag)
        self.wq = nn.Linear(d_key, d_key)
        self.wk = nn.Linear(d_key, d_key)
        self.wv = nn.Linear(d_value, d_value)
        self.wo = nn.Linear(d_value, d_key)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)  # B x T x D
        N = self.n_heads

        query, key, value = (x.split_dim(-1, N).combine_dims(0, -1) for x in (query, key, value))

        #query, key, value = (x.contiguous().view(B, -1, N, D//N).transpose(2, 1).contiguous().view(B*N, -1, D//N)
        #                         for x in (query, key, value))

        outputs = self.attention(query, key, value)  # (B x N) x T x (D//N)
        outputs = outputs.split_dim(0, N).combine_dims(-1, 1)

        return self.wo(outputs)

class EncoderLayer(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.selfattn = ResidualBlock(
            MultiHead2(
                args.d_model, args.d_model, args.n_heads, args.drop_ratio,
                use_wo=args.use_wo),
            args.d_model, args.drop_ratio)
        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden),
            args.d_model, args.drop_ratio)

    def forward(self, x, mask=None):
        return self.feedforward(self.selfattn(x, x, x))

class DecoderLayer(nn.Module):

    def __init__(self, args, causal=True, diag=False):
        super().__init__()
        self.positional = positional
        self.selfattn = ResidualBlock(
            MultiHead(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal, diag),
            args.d_model, args.drop_ratio)

        self.attention = ResidualBlock(
            MultiHead(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio),
            args.d_model, args.drop_ratio)

        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden),
            args.d_model, args.drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)
        x = self.feedforward(self.attention(x, encoding, encoding))
        return x

class Encoder(nn.Module):

    def __init__(self, field, args):
        super().__init__()
        if args.share_embeddings:
            self.out = nn.Linear(args.d_model, len(field.vocab))
        else:
            self.embed = nn.Embedding(len(field.vocab), args.d_model)
        self.layers = nn.ModuleList(
            [EncoderLayer(args) for i in range(args.n_layers)])
        self.dropout = nn.Dropout(args.drop_ratio)
        self.field = field
        self.d_model = args.d_model
        self.share_embeddings = args.share_embeddings

    def forward(self, x):
        if self.share_embeddings:
            x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
        else:
            x = self.embed(x)
        x += positional_encodings_like(x)
        encoding = [x]

        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x)
            encoding.append(x)
        return encoding

class Decoder(nn.Module):

    def __init__(self, field, args, causal=True, diag=False):

        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(args, causal)
            for i in range(args.n_layers)])

        self.out = nn.Linear(args.d_model, len(field.vocab))

        self.dropout = nn.Dropout(args.drop_ratio)
        self.d_model = args.d_model
        self.field = field
        self.length_ratio = args.length_ratio
        self.positional = positional
        self.orderless = args.input_orderless

    def forward(self, x, encoding, input_embeddings=False):

        if not input_embeddings:  # compute input embeddings
            if x.ndimension() == 2:
                x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))
            elif x.ndimension() == 3:  # softmax relaxation
                x = x @ self.out.weight * math.sqrt(self.d_model)  # batch x len x embed_size

        if not self.orderless:
            x += positional_encodings_like(x)
        x = self.dropout(x)

        for l, (layer, enc) in enumerate(zip(self.layers, encoding[1:])):
            x = layer(x, enc)
        return x

class Transformer(nn.Module):

    def __init__(self, src, trg, args):
        super().__init__()
        self.encoder = Encoder(src, args)
        self.decoder = Decoder(trg, args)
        self.field = trg
        self.share_embeddings = args.share_embeddings
        if args.share_embeddings:
            self.encoder.out.weight = self.decoder.out.weight

    def forward(self, encoder_inputs, decoder_inputs, decoding=False, beam=1,
                alpha=0.6, return_probs=False):

        encoding = self.encoder(encoder_inputs)

        if (return_probs and decoding) or (not decoding):
            out = self.decoder(decoder_inputs, encoding)

        if decoding:
            if beam == 1:
                output = self.decoder.greedy(encoding)
            else:
                output = self.decoder.beam_search(encoding, beam, alpha)

            if return_probs:
                return output, out, F.softmax(self.decoder.out(out))
            return output

        if return_probs:
            return out, F.softmax(self.decoder.out(out))
        return out

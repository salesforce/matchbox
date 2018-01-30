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

class FeedForward(nn.Module):

    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.linear1 = Linear(d_model, d_hidden)
        self.linear2 = Linear(d_hidden, d_model)

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

class MultiHead(nn.Module):

    def __init__(self, d_key, d_value, n_heads, drop_ratio,
                causal=False, diag=False):
        super().__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal, diag=diag)
        self.wq = Linear(d_key, d_key)
        self.wk = Linear(d_key, d_key)
        self.wv = Linear(d_value, d_value)
        self.wo = Linear(d_value, d_key)
        self.n_heads = n_heads

    def forward(self, query, key, value, mask=None, weights=None, beta=0, tau=1):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)   # B x T x D
        B, D = query.size(0), query.size(2)
        N = self.n_heads

        query, key, value = (x.contiguous().view(B, -1, N, D//N).transpose(2, 1).contiguous().view(B*N, -1, D//N)
                                for x in (query, key, value))

        outputs = self.attention(query, key, value)  # (B x n) x T x (D/n)
        outputs = outputs.contiguous().view(B, N, -1, D//N).transpose(2, 1).contiguous().view(B, -1, D)

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
        return self.feedforward(self.selfattn(x, x, x, mask))

class DecoderLayer(nn.Module):

    def __init__(self, args, causal=True, diag=False, positional=False):
        super().__init__()
        self.positional = positional
        self.selfattn = ResidualBlock(
            MultiHead(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio, causal, diag),
            args.d_model, args.drop_ratio)

        self.attention = ResidualBlock(
            MultiHead(args.d_model, args.d_model, args.n_heads,
                    args.drop_ratio),  # only noisy when doing cross-attention
            args.d_model, args.drop_ratio)

        if positional:
            self.pos_selfattn = ResidualBlock(
            MultiHead(args.d_model, args.d_model, args.n_heads,   # first try 1 positional head
                    args.drop_ratio, causal, diag),
            args.d_model, args.drop_ratio, pos=2)

        self.feedforward = ResidualBlock(
            FeedForward(args.d_model, args.d_hidden),
            args.d_model, args.drop_ratio)

    def forward(self, x, encoding):
        x = self.selfattn(x, x, x)   #
        if self.positional:
            pos_encoding, weights = positional_encodings_like(x), None
            x = self.pos_selfattn(pos_encoding, pos_encoding, x)  # positional attention
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

    def __init__(self, field, args, causal=True,
                positional=False, diag=False, cosine_output=False):

        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(args, causal, diag, positional)
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

    def denum(self, data, target=True):
        field = self.decoder.field if target else self.encoder.field
        return field.reverse(data.unsqueeze(0))[0]

    def apply_mask(self, inputs, mask, p=1):
        _mask = Variable(mask.long())
        outputs = inputs * _mask + (1 + (-1) * _mask) * p
        return outputs

    def apply_mask_cost(self, loss, mask, batched=False):
        loss.data *= mask
        cost = loss.sum() / (mask.sum() + TINY)

        if not batched:
            return cost

        loss = loss.sum(1, keepdim=True) / (TINY + Variable(mask).sum(1, keepdim=True))
        return cost, loss

    def output_decoding(self, outputs):
        field, text = outputs
        if field is 'src':
            return self.encoder.field.reverse(text.data)
        else:
            return self.decoder.field.reverse(text.data)

    def prepare_sources(self, batch, masks=None):
        masks = self.prepare_masks(batch.src) if masks is None else masks
        return batch.src, masks

    def prepare_inputs(self, batch, inputs=None, distillation=False, masks=None):
        if inputs is None:   # use batch
            if distillation:
                inputs = batch.dec
            else:
                inputs = batch.trg

            decoder_inputs = inputs[:, :-1].contiguous()   # 2D nputes
            decoder_masks = self.prepare_masks(inputs[:, 1:]) if masks is None else masks

        else:  # use student outputs -- manually panding <init>
            if inputs.ndimension() == 2:  # input word indices
                decoder_inputs = Variable(inputs.data.new(inputs.size(0), 1).fill_(self.field.vocab.stoi['<init>']))
                if inputs.size(1) > 1:
                    decoder_inputs = torch.cat((decoder_inputs, inputs[:, :-1]), dim=1)
            else:                         # input one-hot/softmax
                decoder_inputs = Variable(inputs.data.new(inputs.size(0), 1, inputs.size(2))).fill_(0)
                decoder_inputs[:, self.field.vocab.stoi['<init>']] = 1
                if inputs.size(1) > 1:
                    decoder_inputs = torch.cat((decoder_inputs, inputs[:, :-1, :]))

            decoder_masks = self.prepare_masks(inputs) if masks is None else masks
        return decoder_inputs, decoder_masks

    def prepare_targets(self, batch, targets=None, distillation=False, masks=None):
        if targets is None:
            if distillation:
                targets = batch.dec[:, 1:].contiguous()
            else:
                targets = batch.trg[:, 1:].contiguous()
        masks = self.prepare_masks(targets) if masks is None else masks
        return targets, masks

    def prepare_masks(self, inputs):
        if inputs.ndimension() == 2:
            masks = (inputs.data != self.field.vocab.stoi['<pad>']).float()
        else:
            masks = (inputs.data[:, :, self.field.vocab.stoi['<pad>']] != 1).float()
        return masks

    def encoding(self, encoder_inputs, encoder_masks):
        return self.encoder(encoder_inputs, encoder_masks)

    def quick_prepare(self, batch, distillation=False, inputs=None, targets=None,
                        input_masks=None, target_masks=None, source_masks=None):
        inputs,  input_masks   = self.prepare_inputs(batch, inputs, distillation, input_masks)     # prepare decoder-inputs
        targets, target_masks  = self.prepare_targets(batch, targets, distillation, target_masks)  # prepare decoder-targets
        sources, source_masks  = self.prepare_sources(batch, source_masks)
        encoding = self.encoding(sources, source_masks)
        return inputs, input_masks, targets, target_masks, sources, source_masks, encoding, inputs.size(0)

    def forward(self, encoding, encoder_masks, decoder_inputs, decoder_masks,
                decoding=False, beam=1, alpha=0.6, return_probs=False, positions=None, feedback=None):

        if (return_probs and decoding) or (not decoding):
            out = self.decoder(decoder_inputs, encoding, encoder_masks, decoder_masks)

        if decoding:
            if beam == 1:  # greedy decoding
                output = self.decoder.greedy(encoding, encoder_masks, decoder_masks, feedback=feedback)
            else:
                output = self.decoder.beam_search(encoding, encoder_masks, decoder_masks, beam, alpha)

            if return_probs:
                return output, out, softmax(self.decoder.out(out))
            return output

        if return_probs:
            return out, softmax(self.decoder.out(out))
        return out

    def cost(self, decoder_targets, decoder_masks, out=None):
        # get loss in a sequence-format to save computational time.
        decoder_targets, out = mask(decoder_targets, out, decoder_masks.byte())
        logits = self.decoder.out(out)
        loss = F.cross_entropy(logits, decoder_targets)
        return loss

    def batched_cost(self, decoder_targets, decoder_masks, probs, batched=False):
        # get loss in a batch-mode

        if decoder_targets.ndimension() == 2:  # batch x length
            loss = -torch.log(probs + TINY).gather(2, decoder_targets[:, :, None])[:, :, 0]  # batch x length
        else:
            loss = -(torch.log(probs + TINY) * decoder_targets).sum(-1)
        return self.apply_mask_cost(loss, decoder_masks, batched)

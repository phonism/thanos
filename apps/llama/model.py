import sys
sys.path.append('../../python')
import time
import thanos
from thanos import nn, init
import thanos.nn.functional as F
import torch
import numpy as np

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embd):
        super(FeedFoward, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embd, n_head):
        super(TransformerBlock, self).__init__()
        head_size = n_embd // n_head
        self.sa = nn.MultiheadAttention(n_embd, n_head)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        lx = self.ln1(x)
        lx, _ = self.sa(lx)
        x = x + lx
        x = x + self.ffwd(self.ln2(x))
        return x

class Transformer(nn.Module):

    def __init__(self, vocab_size, n_embd=896, block_size=32, n_layer=24, n_head=14):
        super(Transformer, self).__init__()
        self.vocab_size = vocab_size
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(thanos.Tensor(np.arange(T), device=idx.device)) # (T,C)
        pos_emb = F.reshape(pos_emb, (1,) + pos_emb.shape)
        pos_emb = F.broadcast_to(pos_emb, tok_emb.shape)
        x = tok_emb + pos_emb # (B,T,C)
        x = self.blocks(x) # (B,T,C)
        x = self.ln_f(x) # (B,T,C)
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = F.reshape(logits, (B*T, C))
            targets = F.reshape(targets, (B*T,))
            loss = nn.SoftmaxLoss()(logits, targets)

        return logits, loss

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = thanos.Tensor(encode(text), device=thanos.cuda())
n = int(0.9 * len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 32
batch_size = 32

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = F.stack([data[i:i+block_size] for i in ix])
    y = F.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model = Transformer(vocab_size=vocab_size)
model.cuda()
optimizer = thanos.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()
for iter in range(10000):
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    if iter % 100 == 0:
        print("step:", iter, " loss:", loss.detach().numpy(), " time:", time.time() - start_time)
        start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import sys
sys.path.append('../../python')
import thanos
from thanos import nn, init
import time
import thanos.nn.functional as F
import torch
import numpy as np

class Config:
    layer_norm = nn.RMSNorm

class FeedFowardSwiGLU(nn.Module):
    """ 
    SwiGLU: https://arxiv.org/pdf/2002.05202.pdf
    """
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim, bias=False)
        self.down = nn.Linear(hidden_dim, dim, bias=False)
        self.up = nn.Linear(dim, hidden_dim, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.down(self.act(self.gate(x)) * self.up(x))
        return self.dropout(out)

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """

    def __init__(self, n_embed, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embed, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_embed),
            nn.Dropout(0),
        )

    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    """ Transformer block: communication followed by computation """
    def __init__(self, n_embed, n_head):
        super(TransformerBlock, self).__init__()
        head_size = n_embed // n_head
        self.sa = nn.MultiheadAttention(n_embed, n_head)
        self.ffwd = FeedFowardSwiGLU(n_embed, 4 * n_embed)
        #self.ffwd = FeedFoward(n_embed, 4 * n_embed)
        #self.ln1 = nn.LayerNorm(n_embed)
        self.ln1 = nn.RMSNorm(n_embed)
        #self.ln2 = nn.LayerNorm(n_embed)
        self.ln2 = nn.RMSNorm(n_embed)

    def forward(self, x):
        lx = self.ln1(x)
        lx, _ = self.sa(lx)
        x = x + lx
        x = x + self.ffwd(self.ln2(x))
        return x
    
class Transformer(nn.Module):
    """
    transformer
    """
    def __init__(self, vocab_size, n_embed=64, block_size=32, n_layer=4, n_head=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_embed = n_embed
        self.block_size = block_size
        self.n_layer = n_layer
        self.n_head = n_head
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.position_embedding_table = nn.Embedding(block_size, n_embed)
        #self.rotary_emb = nn.RotaryEmbedding(n_embed, block_size)
        self.blocks = nn.Sequential(*[TransformerBlock(n_embed, n_head=n_head) for _ in range(n_layer)])
        #self.ln_f = nn.LayerNorm(n_embed) # final layer norm
        self.ln_f = nn.RMSNorm(n_embed) # final layer norm
        self.lm_head = nn.Linear(n_embed, vocab_size)

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
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = thanos.Tensor(encode(text), device=thanos.cuda())
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

block_size = 32
batch_size = 16

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = F.stack([data[i:i+block_size] for i in ix])
    y = F.stack([data[i+1:i+block_size+1] for i in ix])
    return x, y

model = Transformer(vocab_size=vocab_size, n_embed=512)
model.cuda()
optimizer = thanos.optim.AdamW(model.parameters(), lr=0.001)

print("num_parameters:", model.num_parameters())

start_time = time.time()
total_loss = 0
total_cnt = 0

for iter in range(100):
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    total_loss += loss.detach().numpy()
    total_cnt += 1
    if iter % 100 == 0:
        print("step:", iter, " loss:", total_loss / total_cnt, " time:", time.time() - start_time)
        total_loss = 0
        total_cnt = 0
        start_time = time.time()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

import sys
sys.path.append('../../python')
import json
import thanos
from thanos import nn, init
from thanos.amp import autocast
from thanos.utils import profile
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
        self.sa = nn.FusedMultiheadAttention(n_embed, n_head)
        #self.sa = nn.MultiheadAttention(n_embed, n_head)
        self.ffwd = FeedFowardSwiGLU(n_embed, 4 * n_embed)
        self.ln1 = nn.FusedRMSNorm(n_embed)
        self.ln2 = nn.FusedRMSNorm(n_embed)

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
    def __init__(self, vocab_size, n_embed=64, block_size=32, n_layer=24, n_head=16):
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
        #self.ln_f = nn.FusedLayerNorm(n_embed) # final layer norm
        #self.ln_f = nn.RMSNorm(n_embed) # final layer norm
        self.ln_f = nn.FusedRMSNorm(n_embed) # final layer norm
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

        # TODO 似乎这里反向传播有问题?
        if targets is None:
            targets = idx[:, 1:]  # 将idx错位一个
            logits = logits[:, :-1, :]  # 对应地调整logits

        B, T, C = logits.shape
        logits = F.reshape(logits, (B*T, C))
        targets = F.reshape(targets, (B*T,))
        loss = nn.SoftmaxLoss()(logits, targets)

        return logits, loss

vocab_size = 151936
block_size = 1024

model = Transformer(vocab_size=vocab_size, n_embed=64 * 16, block_size=block_size)
model.cuda()
optimizer = thanos.optim.AdamW(model.parameters(), lr=0.0001)

print("num_parameters:", model.num_parameters())



batch_size = 2

def get_batch(bs):
    x = []
    y = []
    with open("../../../datasets/train_data") as f:
        for line in f:
            js = json.loads(line)
            x.append(js["input_ids"][:1024])
            y.append(js["input_ids"][1:1025])
            if len(x) == batch_size:
                xx = thanos.Tensor(np.array(x), device=thanos.cuda())
                yy = thanos.Tensor(np.array(y), device=thanos.cuda())
                x = []
                y = []
                yield xx, yy

start_time = time.time()
total_cnt = 1
batch_loss = 0
batch_cnt = 0

#model_state_dict, optimizer_state_dict = thanos.load_checkpoint("checkpoints/checkpoint.bin")
#model.load_state_dict(model_state_dict)
#optimizer.load_state_dict(optimizer_state_dict)
print("load done")

accumulation_steps = 64
for idx, data in enumerate(get_batch(batch_size)):
    x = data[0]
    y = data[1]
    with autocast():
        logits, loss = model(x, y)
    loss = loss / accumulation_steps
    loss.backward()
        
    batch_loss += loss.detach().numpy()
    batch_cnt += 1
    if (total_cnt + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
        print("step:", total_cnt, " loss:", batch_loss, " time:", time.time() - start_time)
        batch_loss = 0
        batch_cnt = 0
        start_time = time.time()
    if total_cnt % 2000 == 0:
        thanos.save_checkpoint(model.state_dict(), optimizer.state_dict(), "~/workspace/luqi03/checkpoints/checkpoint.bin")
        print("save done!")
    total_cnt += 1

if (total_cnt + 1) % accumulation_steps != 0:
    optimizer.step()
    optimizer.zero_grad()

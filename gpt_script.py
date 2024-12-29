import torch
import torch.nn as nn
from torch.nn import functional as F

block_size = 8
batch_size = 32
eval_interval = 300
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
eval_iters = 200
max_iters = 10000
n_embed = 32


torch.manual_seed(1337)

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

# stoi and itos are dictionaries that allow us to convert characters to integers and vice versa.
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype = torch.long)

n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters, device=device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

            



class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # Basically, every token has an embedding of size 65
        self.token_embedding_table = nn.Embedding(vocab_size, n_embed)
        self.postion_embedding_table = nn.Embedding(block_size, n_embed)
        self.lm_head = nn.Linear(n_embed, vocab_size)

    def forward(self, idx, targets = None):
        
        token_emb = self.token_embedding_table(idx) # B, T, n_embed
        pos_emb = self.postion_embedding_table(idx) # B, T, n_embed
        emb = token_emb + pos_emb
        logits = self.lm_head(emb) # B, T, vocab_size

        if targets == None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            # For every element in B*T(32 elements), C is the probability distribution of the next character
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


    

model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr = 1e-3)

for steps in range(max_iters):

    if steps%eval_interval == 0:
        losses = estimate_loss()
        print(f'step {steps}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    print(steps, loss.item())


# print(loss.item())

idx = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(idx, max_new_tokens=1000)[0].tolist()))
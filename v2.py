import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 32  # how many independent sequences will we process in parallel?
block_size = 8  # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32
# ----------------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()


# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))  # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]


def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()

    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)

        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    model.train()
    return out


class Head(nn.Module):
    """ One head of self-attention """

    def __init__(self, head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape                 # C = n_embd

        k = self.key(x)                   # (B, T, head_size)
        q = self.query(x)                 # (B, T, head_size)

        # Compute attention scores ("affinities"), scaled by 1/sqrt(head_size)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5   # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)      # (B, T, T)

        # Weighted aggregation of the values
        v = self.value(x)                 # (B, T, head_size)
        out = wei @ v                     # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class MultiHeadAttention(nn.Module):
    """ Multiple Heads of self-attention in parallel """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out) # Linear Transformation of the outcome from concat-ed heads

        return out 
    

class FeedForward(nn.Module):
    """ A simple linear layer followed by a non-linearity then projection"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd), # From paper, inner-layer has dimensionality of 4 times the dimensionality of input and output
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # Projection layer going back to the residual pathway
        )

    def forward(self, x):
        return self.net(x)


class MyLayerNorm:
    """
    LayerNorm adapted from the BatchNorm1d in the makemore series.
    Reference only: the model uses PyTorch's nn.LayerNorm instead.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps

        # Parameters (trained with backprop)
        self.gamma = torch.ones(dim)   # learnable scale (bngain in makemore)
        self.beta = torch.zeros(dim)   # learnable shift (bnbias in makemore)

    def __call__(self, x):
        # Calculate the forward pass
        # BatchNorm averaged over dim 0 (across the batch)
        # LayerNorm averages over the last dim (across features), giving one mean/variance per token.
        xmean = x.mean(-1, keepdim=True)                 # per-token mean over features
        xvar = x.var(-1, keepdim=True, unbiased=False)   # per-token variance over features

        xhat = (x - xmean) / torch.sqrt(xvar + self.eps)  # normalize to zero mean, unit variance
        self.out = self.gamma * xhat + self.beta

        return self.out

    def parameters(self):
        return [self.gamma, self.beta]


class Block(nn.Module):
    """ Transformer Block: Communication followed by Computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.self_attention = MultiHeadAttention(n_head, head_size) # n heads of (n_embd// n_head)-dimensional self-attention
        self.feed_forward = FeedForward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # The residual pathway (also called the residual stream) is the main path that x takes from the embeddings straight to lm_head.
        # Fork off, compute some computation and come back
        """
        token_emb + pos_emb
        │
        x ──────────────┐
        │               ▼
        │           layer_norm_1
        │               ▼
        │        self_attention
        ▼               │
       (+) ◄────────────┘      Block 1
        │
        x ──────────────┐
        │               ▼
        │          layer_norm_2
        │               ▼
        │         feed_forward
        ▼               │
       (+) ◄────────────┘
        │
       ... Block 2, Block 3 ...
        │
     layer_norm_final
        │
     lm_head

        """
        x = x + self.self_attention(self.layer_norm_1(x))
        x = x + self.feed_forward(self.layer_norm_2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd) # (B, T, C)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )
        self.layer_norm_final = nn.LayerNorm(n_embd)

        # Language Modeling Head
        self.lm_head = nn.Linear(n_embd, vocab_size) 


    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B, T) tensor of integers
        token_embedding = self.token_embedding_table(idx)  # (B, T, n_embd)
        pos_embedding = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = token_embedding + pos_embedding
        x = self.blocks(x) # (B, T, C)
        x = self.layer_norm_final(x)
        logits = self.lm_head(x) # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            # B: 32, T: 8, C: 65
            # print(f"B: {B}, T: {T}, C: {C}")

            logits = logits.view(B * T, C)
            # torch.Size([256, 65])
            # print(logits.shape)
            # print(f'logits: {logits}')

            targets = targets.view(B * T)
            # torch.Size([256])
            # print(targets.shape)
            # print(f'targets: {targets}')

            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]

            # get the predictions
            logits, loss = self(idx_cond)

            # focus only on the last time step
            logits = logits[:, -1, :]  # becomes (B, C)

            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1)  # (B, C)

            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)

            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)

        return idx


model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for iter in range(max_iters):  # increase number of steps for good results...
    # every once in awhile evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))

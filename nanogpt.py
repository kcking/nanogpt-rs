import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Tuple, Union
import os
import urllib.request

# hyperparameters
learning_rate = 3e-4
max_iters = 5000
eval_interval = 500
eval_iters = 10
device = "cuda" if torch.has_cuda else "mps" if torch.has_mps else "cpu"
# device = 'cpu'
block_size = 256
batch_size = 64
n_embed = 384
n_layer = 6
n_head = 6
dropout = 0.2
# --------------

torch.manual_seed(1337)

# andrej karpathy nanogpt from scratch: https://www.youtube.com/watch?v=kCc8FmEb1nY
# Initialize data and vocab
url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
file_name = 'data/tinyshakespeare.txt'
os.makedirs('data', exist_ok=True)
if not os.path.exists(file_name):
    urllib.request.urlretrieve(url, file_name)
    print(f'{file_name} downloaded successfully')
with open(file_name, "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

# embedding
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
def encode(s): return [stoi[c] for c in s]  # string -> list of integers


def decode(l): return "".join([itos[i]
                               for i in l])  # list of integers -> string


assert decode(encode("hiii")) == "hiii"

# Split train and test data
data = torch.tensor(encode(text)).to(device)
n = int(0.9 * len(chars))
train_data = data[n:]
val_data = data[n:]

# training data is processed in random chunks or blocks
# first element has no context, so it is not counted as a sample
train_data[: block_size + 1]
x = train_data[:block_size]
y = train_data[1: block_size + 1]
for t in range(block_size):
    context = x[: t + 1]
    target = y[t]
    # print(f"when input is {context}, target is {target}")


# batch size determines how many blocks we stack into a single tensor
def get_batch(train: bool) -> Tuple[torch.Tensor, torch.Tensor]:
    data = train_data if train else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i + block_size] for i in ix])
    y = torch.stack([data[i + 1: i + block_size + 1] for i in ix])
    return x, y


def estimate_loss(model: nn.Module):
    out = {}
    with torch.no_grad():
        model.eval()
        for train in [True, False]:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(train)
                logits, loss = model(X, Y)
                losses[k] = loss.item()
            out["train" if train else "val"] = losses.mean()
        model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size: int) -> None:
        super().__init__()
        self.key = nn.Linear(n_embed, head_size, bias=False)
        self.query = nn.Linear(n_embed, head_size, bias=False)
        self.value = nn.Linear(n_embed, head_size, bias=False)
        self.register_buffer(
            "tril", torch.tril(torch.ones(block_size, block_size)).to(device)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape
        k = self.key(x)  # (B, T, 16)
        q = self.query(x)  # (B, T, 16)
        # affinities between tokens computed by dot-product of Keys and Queries
        wei = (
            q @ k.transpose(-2, -1) * C**-0.5
        )  # (B, T, 16) @ (B, 16, T) --> (B, T, T); normalize by head_size

        # in an encoder block, delete this line and let all nodes talk to each-other
        # filling in future tokens with -inf creates a decoder block, useful for autoregressive models
        wei = wei.masked_fill(
            self.tril[:T, :T] == 0, float("-inf"))  # type: ignore
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHead(nn.Module):
    def __init__(self, num_heads, head_size) -> None:
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embed, n_embed)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # stack the output of each head
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    """Add non-linear "computation" to the network"""

    def __init__(self, n_embed) -> None:
        super().__init__()
        self.net = nn.Sequential(
            # feedforward layer is 4x higher dimension in paper
            nn.Linear(n_embed, n_embed * 4),
            nn.ReLU(),
            # projection layer back into residual pathway
            nn.Linear(n_embed * 4, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block as defined in the 2017 paper"""

    def __init__(self, n_embed: int, n_head: int) -> None:
        super().__init__()
        head_size = n_embed // n_head
        self.sa = MultiHead(n_head, head_size)
        self.ffwd = FeedForward(n_embed)
        self.ln1 = nn.LayerNorm(n_embed)
        self.ln2 = nn.LayerNorm(n_embed)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embed).to(device)
        self.position_embedding_table = nn.Embedding(
            block_size, n_embed).to(device)
        # self-attention head
        self.blocks = nn.Sequential(
            *[Block(n_embed, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embed)
        # language modelling head
        self.lm_head = nn.Linear(n_embed, vocab_size).to(device)

    def forward(self, idx: torch.Tensor, targets: Union[torch.Tensor, None] = None):
        B, T = idx.shape
        # predict based on idx
        token_emdedding: torch.Tensor = self.token_embedding_table(
            idx
        )  # (Batch, Time, Channels)
        # pass in position in block
        position_embedding = self.position_embedding_table(
            torch.arange(T).to(device)
        )  # (T, C)
        x = token_emdedding + position_embedding  # (B, T, C)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens) -> torch.Tensor:
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus on last time step
            logits = logits[:, -1, :]  # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=1)  # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)  # (B, T+1)
        return idx


m = BigramLanguageModel().to(device)


def generate():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))


if __name__ == "main":
    optim = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    loss = torch.Tensor([0])
    # train
    for iter in range(max_iters):
        print(iter)
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        if iter % 50 == 0:
            generate()
        # sample a batch of data
        (
            xb,
            yb,
        ) = get_batch(True)

        # evaluate loss
        logits, loss = m(xb, yb)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        optim.step()

    generate()

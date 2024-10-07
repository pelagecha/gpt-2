from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import transformers
import tiktoken
import inspect 
# ---------------------------




class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn  = nn.Linear(config.n_embd, 3*config.n_embd) # key query value projections for all heads, but in a batch
        self.c_proj = nn.Linear(config.n_embd, config.n_embd) # output projection
        self.c_proj.GPT_SCALE_INIT = 1 # NOTE to prevent increased variance related to accumulation of residual connections
        # regularization
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimension
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, num_heads, T, head_size)

        # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = (att @ v).transpose(1, 2).contiguous().view(B,T,C) # use Flash attention instead! because it's faster
        
        y = F.scaled_dot_product_attention(q,k,v, is_causal=True) # NOTE Calculates Flash Attention, which uses Kernel Fusion to merge the operations into the same kernel, resulting in memory speedup and calculates SoftMax in stream manner.

        y = y.transpose(1, 2).contiguous().view(B,T,C)
        y = self.c_proj(y) # output projection
        return y


@dataclass
class GPTConfig:
    block_size: int = 1024  # 256
    vocab_size: int = 50257 # 65 # 50,000 BPE merges, 256 bytes tokens, 1 end of text token
    n_layer:    int = 12    # 6
    n_head:     int = 12    # 6
    n_embd:     int = 768   # 384


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu    = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.GPT_SCALE_INIT = 1 # NOTE to prevent increased variance related to accumulation of residual connections

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x
    

class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)       # layernorm 1
        self.attn = CausalSelfAttention(config)  # self-attention
        self.ln_2 = nn.LayerNorm(config.n_embd)       # layernorm 2
        self.mlp = MLP(config)                        # just an MLP

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # apply layernorm 1 with a residual connection and pass through attention
        x = x + self.mlp(self.ln_2(x))       # apply layernorm 2 and pass through MLP
        return x
    


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),                   # weigths of token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd),                   # weights of the position embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),      # list of `n_layer` hidden layers
            ln_f = nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight sharing scheme IMPORTANT!
        self.transformer.wte.weight = self.lm_head.weight # NOTE prevent the bug where this layer is doubled (top and bottom layers are the same)
        self.apply(self._init_weights) # init params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "GPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5 # NOTE 2 blocks (attention, MLP), normalises addup from residual connections
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # NOTE Following Xavier Initialisation: 1/sqrt(num_features)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # override default bias init
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot forward sequence of length {T}, block size is only {self.config.block_size}"
        # forward the token and posisition embeddings
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb
        # forward the blocks of the transformer
        for block in self.transformer.h:
            x = block(x)
        # forward the final layernorm and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimisers(self, weight_decay, learning_rate, device):
        # start with all of the candidate parameters (that require grad)
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device == "cuda"
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused) # Use Fused AdamW
        return optimizer
    

# =========== Generation ===========
import time
if torch.cuda.is_available():
    device = "cuda"
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using {device}")

class DataLoader:
    def __init__(self, B, T):
        self.B = B
        self.T = T
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding("gpt2")
        tokens = enc.encode(text)
        self.tokens = torch.tensor(tokens)
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"1 epoch = {len(self.tokens) // (B*T)} batches")

        self.current_position = 0
    
    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B,T)
        y = (buf[1:]).view(B,T)
        self.current_position += B*T
        if self.current_position + (B*T+1) > len(self.tokens):
            self.current_position = 0
        return x,y

# import tiktoken
# enc = tiktoken.get_encoding("gpt2")
# with open("input.txt", "r") as f:
#     text = f.read()

# text = text[:1000]
# tokens = enc.encode(text)
# B, T = 4,32
# buf = torch.tensor(tokens[:B*T + 1])
# buf = buf.to(device)
# x = buf[:-1].view(B,T)
# y = buf[1:].view(B,T)


# Simulate Large Batch sizes
total_batch_size = 524288
B = 16
T = 1024
assert total_batch_size % (B * T) == 0, "make sure total_batch_size is divisible by B * T!"
grad_accum_steps = total_batch_size // (B * T)
print(f"total desired batch size: {total_batch_size}")
print(f"=> calculated gradient accumulation steps: {grad_accum_steps}")


train_loader = DataLoader(B=4,T=1024) # batch size and sequence length
torch.set_float32_matmul_precision("high") # NOTE Improves the operation performance through decreased precision

model = GPT(GPTConfig(vocab_size=50304)) # NOTE Introduce a "nice" number (padded fom 50257) to improve speed of computation (divisible by 2), so the CUDA Block Tiles run better
model.to(device)
print("Compiling the model...")
model = torch.compile(model) # NOTE Compiling the model takes longer to start, but drastically improves the training time
# optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.95), eps=1e-8)
optimiser = model.configure_optimisers(weight_decay = 0.1, learning_rate = 6e-4, device = device)
max_lr = 6e-4
min_lr = max_lr * 0.1
max_steps = 50
warmup_steps = max_steps / 5

def get_lr(it): # Learning Rate scheduler
    if it < warmup_steps: # Warmup region
        return max_lr * (it+1) / warmup_steps 
    if it > max_steps:
        return min_lr
    # Cosine decay
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # starts at 0 and goes to 0
    return min_lr + coeff * (max_lr - min_lr)

for step in range(max_steps):
    torch.cuda.synchronize()
    t0 = time.time()
    optimiser.zero_grad()
    loss_accum = 0.0

    for micro_step in range(grad_accum_steps): # simulating large batches
        x,y = train_loader.next_batch()
        x,y = x.to(device), y.to(device)
        with torch.autocast(device_type=device, dtype=torch.bfloat16): # NOTE Autocasts to bfloat16 (same exponent as float32, but smaller mantissa) that doesn't require gradient scaler
            logits, loss = model(x, y)
        loss = loss / grad_accum_steps # reintroduces mean into Sq error
        loss_accum += loss.detach()
        loss.backward()

    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping (prevents sudded gradient bursts)

    lr = get_lr(step)
    for param_group in optimiser.param_groups:
        param_group["lr"] = lr
    optimiser.step()

    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000

    tokens_processed = train_loader.B * train_loader.T * grad_accum_steps
    tokens_per_sec = tokens_processed / dt

    print(f"step {step} | loss: {loss_accum.item()} | lr: {lr:.6f} | norm: {norm:.4f} | dt: {dt:.2f}ms")


import sys; sys.exit(0)


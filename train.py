from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import transformers
import tiktoken
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

    @classmethod
    def from_pretrained(cls, model_type):
        # ================================================================
        # =-=-=-=-=-=-=- loading weights from huggingface -=-=-=-=-=-=-=-=
        # ================================================================
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them
        assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
        for k in sd_keys_hf:
            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    

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

train_loader = DataLoader(B=4,T=1024) # batch size and sequence length
torch.set_float32_matmul_precision("high") # NOTE Improves the operation performance through decreased precision

model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
print("Compiling the model...")
model = torch.compile(model) # NOTE Compiling the model takes longer to start, but drastically improves the training time

optimiser = torch.optim.AdamW(model.parameters(), lr=3e-4)
for i in range(50):
    torch.cuda.synchronize()
    t0 = time.time()

    x,y = train_loader.next_batch()
    x,y = x.to(device), y.to(device)
    optimiser.zero_grad()
    with torch.autocast(device_type=device, dtype=torch.bfloat16): # NOTE Autocasts to bfloat16 (same exponent as float32, but smaller mantissa) that doesn't require gradient scaler
        logits, loss = model(x, y)
    loss.backward()
    optimiser.step()
    torch.cuda.synchronize()
    t1 = time.time()
    dt = (t1-t0)*1000
    print(f"step {i}, loss: {loss.item()}, dt: {dt:.2f}ms")



import sys; sys.exit(0)


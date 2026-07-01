# GPT-2 From Scratch: Line-by-Line Code Explanation

This document explains **what every line of code does**, **why it exists**, **its syntax**, and **what values flow through it**.

---

# Imports

```python
from dataclasses import dataclass
```

### Purpose

Used to create a configuration class without manually writing constructors.

Instead of:

```python
class GPTConfig:
    def __init__(self):
        ...
```

we can write:

```python
@dataclass
class GPTConfig:
```

and Python automatically creates:

```python
__init__()
__repr__()
__eq__()
```

for us.

---

```python
import math
```

Used for mathematical functions.

Example:

```python
math.sqrt(64)
```

returns:

```python
8
```

Used in attention scaling.

---

```python
import torch
```

Imports PyTorch.

Provides:

```python
Tensor
GPU support
Autograd
```

---

```python
import torch.nn as nn
```

Provides neural network layers.

Examples:

```python
nn.Linear
nn.LayerNorm
nn.Embedding
```

---

```python
from torch.nn import functional as F
```

Provides functions that don't require state.

Examples:

```python
F.softmax()
F.relu()
F.cross_entropy()
```

---

# CausalAttention Class

---

```python
class CausalAttention(nn.Module):
```

Creates a custom PyTorch layer.

Equivalent to saying:

```text
Build a reusable Attention module.
```

---

# Constructor

```python
def __init__(self, config):
```

Runs once when creating the model.

Example:

```python
attention = CausalAttention(config)
```

---

```python
super().__init__()
```

Calls the parent class constructor.

Required when inheriting from:

```python
nn.Module
```

---

```python
assert config.n_embd % config.n_head == 0
```

Checks:

```python
768 % 12 == 0
```

Result:

```python
True
```

Why?

Each head must get equal dimensions.

---

# QKV Projection

```python
self.c_attn = nn.Linear(
    config.n_embd,
    3 * config.n_embd
)
```

For GPT-2:

```python
nn.Linear(
    768,
    2304
)
```

Creates one matrix that computes:

```text
Query
Key
Value
```

simultaneously.

Input:

```python
(B,T,768)
```

Output:

```python
(B,T,2304)
```

---

# Output Projection

```python
self.c_proj = nn.Linear(
    config.n_embd,
    config.n_embd
)
```

Equivalent:

```python
768 → 768
```

Used after attention.

Combines information from all heads.

---

```python
self.n_head = config.n_head
```

Stores:

```python
12
```

---

```python
self.n_embd = config.n_embd
```

Stores:

```python
768
```

---

# Register Buffer

```python
self.register_buffer(
    "bias",
```

Registers a tensor inside the model.

Difference:

| Parameter | Buffer |
|------------|---------|
| Trained | No |
| Saved | Yes |
| Gradient | No |

---

```python
torch.tril(
    torch.ones(
        config.block_size,
        config.block_size
    )
)
```

Creates:

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

Lower triangular matrix.

---

```python
.view(
    1,
    1,
    block_size,
    block_size
)
```

Adds dimensions.

Shape:

```python
(1,1,1024,1024)
```

Makes broadcasting easier.

---

# Forward Pass

```python
def forward(self, x):
```

Receives:

```python
(B,T,C)
```

Example:

```python
(5,8,768)
```

---

```python
B, T, C = x.size()
```

Extracts dimensions.

Example:

```python
B = 5
T = 8
C = 768
```

---

# Generate QKV

```python
qkv = self.c_attn(x)
```

Shape:

```python
(5,8,2304)
```

---

```python
q,k,v = qkv.split(
    self.n_embd,
    dim=2
)
```

Splits:

```text
2304

↓

768
768
768
```

Each shape:

```python
(5,8,768)
```

---

# Multi-Head Reshape

```python
k.view(
    B,
    T,
    self.n_head,
    C // self.n_head
)
```

Substitute:

```python
(5,8,12,64)
```

---

```python
.transpose(1,2)
```

Swaps:

```python
T ↔ Heads
```

Shape:

```python
(5,12,8,64)
```

Done for:

```python
q
k
v
```

---

# Attention Matrix

```python
k.transpose(-2,-1)
```

Before:

```python
(5,12,8,64)
```

After:

```python
(5,12,64,8)
```

---

```python
att = q @ k.transpose(-2,-1)
```

Matrix multiplication.

Shape:

```python
(5,12,8,8)
```

Meaning:

```text
Token-to-token similarities
```

---

# Scaling

```python
* (1.0 / math.sqrt(
    k.size(-1)
))
```

Head size:

```python
64
```

Scale:

```python
1/8
```

Purpose:

Prevent huge values.

---

# Causal Mask

```python
att.masked_fill(
    self.bias[:,:,:T,:T] == 0,
    float('-inf')
)
```

Future positions become:

```text
-infinity
```

After softmax:

```text
Probability = 0
```

---

# Softmax

```python
att = F.softmax(
    att,
    dim=-1
)
```

Converts scores:

```text
[-2,1,5]
```

into probabilities:

```text
[0.001,
 0.018,
 0.981]
```

---

# Weighted Value Sum

```python
y = att @ v
```

Shape:

```python
(5,12,8,64)
```

Applies attention weights.

---

# Merge Heads

```python
y.transpose(1,2)
```

Shape:

```python
(5,8,12,64)
```

---

```python
.contiguous()
```

Ensures memory layout is continuous.

Required before:

```python
view()
```

---

```python
.view(B,T,C)
```

Shape:

```python
(5,8,768)
```

Heads merged back together.

---

```python
y = self.c_proj(y)
```

Final projection.

Output:

```python
(5,8,768)
```

---

# MLP Class

```python
class MLP(nn.Module):
```

Feed Forward Network.

---

```python
self.c_fc = nn.Linear(
    config.n_embd,
    4 * config.n_embd
)
```

GPT-2:

```python
768 → 3072
```

---

```python
self.gelu = nn.GELU(
    approximate='tanh'
)
```

Activation function.

More modern than ReLU.

---

```python
self.c_proj = nn.Linear(
    3072,
    768
)
```

Projects back down.

---

# MLP Forward

```python
x = self.c_fc(x)
```

Shape:

```python
(B,T,3072)
```

---

```python
x = self.gelu(x)
```

Adds non-linearity.

---

```python
x = self.c_proj(x)
```

Shape:

```python
(B,T,768)
```

---

# Block Class

```python
class Block(nn.Module):
```

One Transformer layer.

---

```python
self.ln_1
```

LayerNorm before attention.

---

```python
self.attn
```

Attention layer.

---

```python
self.ln_2
```

LayerNorm before MLP.

---

```python
self.mlp
```

Feed-forward network.

---

# Block Forward

```python
x = x + self.attn(
    self.ln_1(x)
)
```

Flow:

```text
x
 ↓
LayerNorm
 ↓
Attention
 ↓
Residual Add
```

---

```python
x = x + self.mlp(
    self.ln_2(x)
)
```

Flow:

```text
x
 ↓
LayerNorm
 ↓
MLP
 ↓
Residual Add
```

---

# GPTConfig

```python
@dataclass
class GPTConfig:
```

Stores:

```python
block_size
vocab_size
n_layer
n_head
n_embd
```

---

# GPT Class

```python
class GPT(nn.Module):
```

Entire model.

---

# Embeddings

```python
wte
```

Word Token Embedding.

Shape:

```python
(50257,768)
```

---

```python
wpe
```

Word Position Embedding.

Shape:

```python
(1024,768)
```

---

# Transformer Blocks

```python
h = nn.ModuleList([
    Block(config)
    for _ in range(
        config.n_layer
    )
])
```

Creates:

```python
12 Blocks
```

Stores them in a list.

---

# Final LayerNorm

```python
ln_f
```

Applied after all blocks.

---

# LM Head

```python
self.lm_head
```

Linear:

```python
768 → 50257
```

Produces logits.

---

# GPT Forward

```python
B,T = idx.size()
```

Example:

```python
(5,8)
```

---

```python
assert T <= block_size
```

Ensures sequence isn't too long.

---

```python
pos = torch.arange(
    0,
    T
)
```

Creates:

```text
[0,1,2,3,4,5,6,7]
```

---

```python
pos_emb = self.transformer.wpe(pos)
```

Shape:

```python
(8,768)
```

---

```python
tok_emb = self.transformer.wte(idx)
```

Shape:

```python
(5,8,768)
```

---

```python
x = tok_emb + pos_emb
```

Shape:

```python
(5,8,768)
```

---

# Run Transformer

```python
for block in self.transformer.h:
    x = block(x)
```

Runs through all 12 layers.

---

```python
x = self.transformer.ln_f(x)
```

Final LayerNorm.

---

```python
logits = self.lm_head(x)
```

Shape:

```python
(5,8,50257)
```

---

```python
return logits
```

Returns raw scores.

Not probabilities yet.

---

# Generation Loop

```python
while x.size(1) < max_length:
```

Generate until:

```python
30 tokens
```

---

```python
logits = model(x)
```

Forward pass.

---

```python
logits = logits[:,-1,:]
```

Take last token only.

Shape:

```python
(5,50257)
```

---

```python
probs = F.softmax(
    logits,
    dim=-1
)
```

Convert to probabilities.

---

```python
topk_probs,
topk_indices
=
torch.topk(
    probs,
    50
)
```

Keep only top 50 tokens.

---

```python
ix = torch.multinomial(
    topk_probs,
    1
)
```

Randomly sample.

---

```python
xcol = torch.gather(
    topk_indices,
    -1,
    ix
)
```

Retrieve actual token ID.

---

```python
x = torch.cat(
    (x,xcol),
    dim=1
)
```

Append generated token.

Sequence grows:

```text
8 tokens

↓

9 tokens

↓

10 tokens

↓

...
```

---

# Final Mental Model

```text
Input Tokens
      ↓
Embeddings
      ↓
Attention
      ↓
Residual Stream
      ↓
MLP
      ↓
Residual Stream
      ↓
Repeat 12 Times
      ↓
Vocabulary Scores
      ↓
Softmax
      ↓
Next Token
```

Every line in the code contributes to one of these four jobs:

1. Convert tokens into vectors
2. Let tokens communicate (Attention)
3. Process information (MLP)
4. Predict the next token
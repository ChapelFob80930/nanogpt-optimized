# GPT-2 From Scratch: Complete Architecture, Data Flow, and Code Documentation

## Overview

This document explains:

1. How input text flows through GPT-2
2. What every major component does
3. Tensor shapes at every stage
4. The purpose of every line of code
5. How token generation works
6. The intuition behind Attention, MLPs, LayerNorm, and Residual Connections

---

# 1. High-Level Architecture

GPT-2 is a Decoder-Only Transformer.

```text
Input Text
     │
     ▼
Tokenizer (BPE)
     │
     ▼
Token IDs
     │
     ▼
Token Embeddings
+
Position Embeddings
     │
     ▼
Transformer Block × 12
     │
     ▼
Final LayerNorm
     │
     ▼
Linear Projection
     │
     ▼
Vocabulary Logits
     │
     ▼
Softmax
     │
     ▼
Next Token Prediction
```

GPT-2 Small (124M Parameters)

| Parameter | Value |
|------------|--------|
| Vocab Size | 50,257 |
| Context Length | 1024 |
| Layers | 12 |
| Heads | 12 |
| Embedding Dimension | 768 |
| Parameters | ~124M |

---

# 2. Input Flow Example

Prompt:

```text
Hello, I'm a language model,
```

---

## Step 1: Tokenization

```python
tokens = enc.encode("Hello, I'm a language model,")
```

Example output:

```text
[15496, 11, 314, 1101, 257, 3303, 2746, 11]
```

Shape:

```python
(T,)
```

Where:

```text
T = 8
```

---

## Step 2: Create Batch

```python
tokens = tokens.unsqueeze(0)
```

Shape:

```python
(1,8)
```

Create 5 copies:

```python
tokens = tokens.repeat(5,1)
```

Shape:

```python
(B,T)

(5,8)
```

---

# 3. GPTConfig

```python
@dataclass
class GPTConfig:
```

Stores all model hyperparameters.

```python
block_size = 1024
```

Maximum sequence length.

```python
vocab_size = 50257
```

Number of possible tokens.

```python
n_layer = 12
```

Number of Transformer blocks.

```python
n_head = 12
```

Number of attention heads.

```python
n_embd = 768
```

Embedding size.

---

# 4. Embedding Layers

## Token Embedding

```python
wte = nn.Embedding(
    vocab_size,
    n_embd
)
```

Shape:

```text
(50257,768)
```

Meaning:

Each token ID maps to a learned vector.

Example:

```text
Token 15496

↓

[0.12, -0.55, ..., 0.31]
```

768 numbers.

---

## Position Embedding

```python
wpe = nn.Embedding(
    block_size,
    n_embd
)
```

Shape:

```text
(1024,768)
```

Stores positional information.

---

# 5. Forward Pass Through Embeddings

Create position indices:

```python
pos = torch.arange(0,T)
```

Example:

```text
[0,1,2,3,4,5,6,7]
```

---

Position Embeddings:

```python
pos_emb = self.transformer.wpe(pos)
```

Shape:

```python
(8,768)
```

---

Token Embeddings:

```python
tok_emb = self.transformer.wte(idx)
```

Shape:

```python
(5,8,768)
```

---

Combine:

```python
x = tok_emb + pos_emb
```

Broadcasting:

```python
(5,8,768)
+
(8,768)

=
(5,8,768)
```

Each token now knows:

- What word it is
- Where it is

---

# 6. Transformer Block

Each block contains:

```text
Block
├── LayerNorm
├── Self Attention
├── Residual Connection
├── LayerNorm
├── MLP
└── Residual Connection
```

GPT-2 Small contains:

```text
12 Transformer Blocks
```

---

# 7. Residual Stream

Input:

```python
x
```

Shape:

```python
(B,T,768)
```

This tensor is called the:

```text
Residual Stream
```

Think of it as:

```text
The main information highway
```

Every layer:

```text
Reads from it
Adds information
Passes it forward
```

---

## Why Residual Connections Exist

Without residuals:

```text
Layer1
 ↓
Layer2
 ↓
Layer3
```

Gradients must travel through everything.

Can cause:

```text
Vanishing Gradients
```

---

With residuals:

```text
x --------------------+
                      |
Attention ------------+
                      |
new x
```

Gradient has shortcut paths.

Training becomes much easier.

---

# 8. LayerNorm

Code:

```python
self.ln_1 = nn.LayerNorm(768)
```

Formula:

```text
x̂ = (x - mean)/std
```

Then:

```text
y = γx̂ + β
```

Learnable:

```text
γ = scale
β = shift
```

Purpose:

```text
Keeps activations stable
```

---

# 9. Self-Attention Overview

Input:

```python
(B,T,768)
```

Example:

```python
(5,8,768)
```

Goal:

```text
Allow tokens to communicate with each other
```

---

# 10. QKV Projection

Code:

```python
self.c_attn = nn.Linear(
    768,
    3*768
)
```

Shape:

```text
768 → 2304
```

Output:

```python
qkv
```

Shape:

```python
(5,8,2304)
```

---

Split:

```python
q,k,v = qkv.split(
    self.n_embd,
    dim=2
)
```

Each:

```python
(5,8,768)
```

---

Meaning:

### Query

```text
What information am I looking for?
```

### Key

```text
What information do I contain?
```

### Value

```text
What information can I provide?
```

---

# 11. Multi-Head Split

GPT-2:

```text
n_head = 12
```

Head Size:

```text
768 / 12 = 64
```

Reshape:

```python
q.view(
    B,T,12,64
)
```

Shape:

```python
(5,8,12,64)
```

Transpose:

```python
.transpose(1,2)
```

Shape:

```python
(5,12,8,64)
```

Interpretation:

```text
5 batches
12 heads
8 tokens
64 features per head
```

Same process for:

```python
q
k
v
```

---

# 12. Attention Scores

Code:

```python
att = q @ k.transpose(-2,-1)
```

Shapes:

```python
(5,12,8,64)
@
(5,12,64,8)

=
(5,12,8,8)
```

Meaning:

```text
Every token compares itself
to every other token
```

Produces:

```text
T × T matrix
```

For T = 8:

```text
8 × 8
```

---

# 13. Scaling

Code:

```python
att *= 1.0 / math.sqrt(
    k.size(-1)
)
```

Head Size:

```text
64
```

Scale:

```text
1/√64 = 1/8
```

Purpose:

```text
Prevent extremely large dot products
```

Without scaling:

```text
Softmax becomes unstable
```

---

# 14. Causal Mask

Buffer:

```python
torch.tril(
    torch.ones(
        1024,
        1024
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

Apply:

```python
att.masked_fill(
    mask==0,
    -inf
)
```

Future tokens become impossible.

Example:

```text
Token 3
cannot see
Token 4
```

Prevents cheating.

---

# 15. Softmax

```python
att = F.softmax(
    att,
    dim=-1
)
```

Example:

```text
[-2,1,5]
```

Becomes:

```text
[0.001,
 0.018,
 0.981]
```

Interpretation:

```text
Attention probabilities
```

---

# 16. Weighted Sum of Values

```python
y = att @ v
```

Shape:

```python
(5,12,8,64)
```

Each token now contains:

```text
Contextual information
from previous tokens
```

---

# 17. Merge Heads

Before:

```python
(5,12,8,64)
```

Transpose:

```python
(5,8,12,64)
```

Flatten:

```python
(5,8,768)
```

All head outputs are combined.

---

# 18. Output Projection

Code:

```python
self.c_proj
```

Linear:

```text
768 → 768
```

Purpose:

```text
Mix information
from all heads
```

---

# 19. MLP

Code:

```python
768
 ↓
3072
 ↓
768
```

Implemented as:

```python
self.c_fc
```

```python
768 → 3072
```

---

Activation:

```python
GELU
```

---

Projection:

```python
self.c_proj
```

```python
3072 → 768
```

---

Shape Flow:

```python
(B,T,768)

↓

(B,T,3072)

↓

(B,T,768)
```

---

Intuition:

```text
Attention = Communication

MLP = Thinking
```

Attention gathers information.

MLP processes information.

---

# 20. Complete Transformer Block

Code:

```python
x = x + self.attn(
    self.ln_1(x)
)

x = x + self.mlp(
    self.ln_2(x)
)
```

This is called:

```text
Pre-LayerNorm Transformer
```

Process:

```text
Input
 ↓
LayerNorm
 ↓
Attention
 ↓
Residual Add
 ↓
LayerNorm
 ↓
MLP
 ↓
Residual Add
```

---

# 21. Full Transformer Stack

Code:

```python
for block in self.transformer.h:
    x = block(x)
```

Runs:

```text
Block 1
Block 2
Block 3
...
Block 12
```

Shape always remains:

```python
(B,T,768)
```

---

# 22. Final LayerNorm

```python
x = self.transformer.ln_f(x)
```

Shape:

```python
(B,T,768)
```

Purpose:

```text
Final stabilization
before prediction
```

---

# 23. Language Modeling Head

Code:

```python
self.lm_head
```

Linear:

```text
768 → 50257
```

Output:

```python
(B,T,50257)
```

Example:

```python
(5,8,50257)
```

Meaning:

```text
For every token

Predict probability
of every vocabulary token
```

---

# 24. Next Token Generation

Take last position:

```python
logits = logits[:,-1,:]
```

Shape:

```python
(5,50257)
```

---

Convert to probabilities:

```python
probs = F.softmax(
    logits,
    dim=-1
)
```

Shape:

```python
(5,50257)
```

---

Top-K Sampling

```python
topk_probs,
topk_indices
=
torch.topk(
    probs,
    50
)
```

Shape:

```python
(5,50)
```

---

Sample:

```python
ix = torch.multinomial(
    topk_probs,
    1
)
```

Shape:

```python
(5,1)
```

---

Get token:

```python
xcol = torch.gather(
    topk_indices,
    -1,
    ix
)
```

Shape:

```python
(5,1)
```

---

Append:

```python
x = torch.cat(
    (x,xcol),
    dim=1
)
```

Shape:

```python
(5,8)

↓

(5,9)

↓

(5,10)

↓

...
```

Until:

```python
(5,30)
```

---

# 25. Complete Shape Flow

```text
Input IDs
(5,8)

↓

Token Embedding
(5,8,768)

+

Position Embedding
(8,768)

↓

(5,8,768)

↓

Transformer Block ×12

↓

(5,8,768)

↓

Final LayerNorm

↓

(5,8,768)

↓

LM Head

↓

(5,8,50257)

↓

Take Last Token

↓

(5,50257)

↓

Softmax

↓

Sample Token

↓

Append Token

↓

Repeat
```

---

# 26. Mental Model of GPT-2

Think of GPT-2 as three systems:

## 1. Embeddings

```text
Token IDs
↓

Dense Vectors
```

Convert tokens into learnable representations.

---

## 2. Attention

```text
Tokens communicate
with each other
```

Determines:

```text
Who should listen to whom
```

---

## 3. MLP

```text
Processes information
```

Learns:

```text
Patterns
Rules
Reasoning
Concepts
```

---

# Final Intuition

The Residual Stream is the highway.

```text
Embeddings
     ↓
Residual Stream
     ↓
Attention adds information
     ↓
Residual Stream
     ↓
MLP adds information
     ↓
Residual Stream
     ↓
Attention adds more information
     ↓
Residual Stream
     ↓
...
     ↓
Prediction
```

Every layer contributes information to the same stream.

Attention decides:

```text
Where information comes from
```

MLP decides:

```text
What to do with that information
```

After 12 layers, GPT-2 has enough contextual understanding to predict the next token.

If you completely understand:

- Q, K, V
- Multi-Head Attention
- Residual Connections
- LayerNorm
- MLP
- LM Head

then you understand roughly 90% of GPT-2 at the implementation level.
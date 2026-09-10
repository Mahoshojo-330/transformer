# My roadmap for building a tiny transformer

## What I am building

I am not trying to reproduce ChatGPT. I am building a tiny language model that
is small enough to inspect and train locally. Its purpose is to make transformer
mechanics concrete.

The finished learning model will do four things:

1. learn to predict the next token in a small text corpus;
2. generate simple text;
3. show its real queries, keys, values, and attention weights;
4. demonstrate why a KV cache makes generation faster.

It will also run on both the CPU and my Apple GPU so I can learn what hardware
acceleration changes—and what it does not change.

## The model's path

```text
text
  -> token IDs
  -> token embeddings + position embeddings
  -> causal self-attention
  -> residual connection + normalization
  -> feed-forward network
  -> residual connection + normalization
  -> vocabulary scores
  -> next-token probabilities
```

The model repeats the middle “transformer block” one or two times. Keeping it
small lets me print and understand every tensor.

## Meaning, position, and context

A token starts with a reusable word embedding, but its query and key are not
permanently attached to the word:

```text
x_i = token_embedding(token_i) + position_embedding(i)
q_i = x_i @ W_Q
k_i = x_i @ W_K
v_i = x_i @ W_V
```

The matrices `W_Q`, `W_K`, and `W_V` are shared model parameters. Training
changes these parameters. During a forward pass, each token occurrence uses
them to calculate its own `Q`, `K`, and `V`.

In deeper layers, `x_i` has already absorbed information from other tokens.
Therefore the representation of `bank` can differ in `river bank` and `money
bank`, even though both began from the same token embedding.

Attention is not merely a dictionary measure of word similarity. A head can
learn relevance based on meaning, grammar, position, references, or other
patterns useful for next-token prediction.

## Why training is manageable

The corpus, vocabulary, context window, and model will all be tiny. Training is
only meant to prove that the system learns and to give us real tensors to
inspect. It may memorize the small corpus, and that is acceptable for this
project. Broad language ability would require vastly more data and computation.

## Using my GPU

My MacBook Pro has an Apple M5 Pro GPU and unified memory. In PyTorch, the Apple
GPU is selected with the `mps` device (Metal Performance Shaders), not `cuda`:

```python
device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

The transformer mathematics and model code remain the same. The device tells
PyTorch where tensor operations such as matrix multiplication should run.

The initial command-line interface should support:

```text
python train.py --device auto
python train.py --device cpu
python train.py --device mps
```

GPU use does not guarantee a speedup. Our first model is intentionally tiny,
and dispatching work to a GPU has overhead. The CPU may win at that size. We
will therefore keep:

- a tiny configuration for understanding and inspecting the model;
- a somewhat larger configuration for comparing CPU and GPU throughput.

The comparison should include warm-up steps, MPS synchronization before timing,
step time, and tokens processed per second. We will start with ordinary
32-bit floating point. Mixed precision can come later, after correctness and
only if measurement shows it helps.

CPU execution remains valuable as a correctness reference. MPS results may
differ slightly because floating-point operations can be accumulated in a
different order, so tests should use numerical tolerances.

Official references: [PyTorch MPS backend
notes](https://docs.pytorch.org/docs/stable/notes/mps.html) and [MPS environment
variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html).

## How I will work through it

### Stage 1: data

Turn words into integer token IDs and create examples where the target is the
next token. I should be able to inspect one batch by hand.

### Stage 2: attention

Implement the exact calculation:

```text
scores  = Q @ K.T / sqrt(key_dimension)
weights = softmax(masked scores)
output  = weights @ V
```

The causal mask prevents a token from reading future tokens.

### Stage 3: transformer block

Add residual connections, layer normalization, and the feed-forward network.
These make attention part of a complete transformer block rather than the
entire model by itself.

### Stage 4: training

Use next-token cross-entropy loss and automatic differentiation. The first
goal is simply to observe the loss decrease and confirm that gradients reach
the embeddings and `Q/K/V` matrices.

### Stage 5: hardware acceleration

Add one device-selection function and move the model plus batches to CPU or
MPS. Verify one equivalent training step on each backend. Then benchmark both
with a tiny teaching configuration and a larger workload configuration.

### Stage 6: inspection and generation

Generate tokens one at a time and add a command that prints the tensors for a
selected layer and attention head.

### Stage 7: KV cache

First generate by recomputing the whole prefix. Then cache each previous
token's keys and values. A new token still calculates fresh attention weights:

```text
new query + cached keys -> new attention weights
new attention weights + cached values -> new output
```

The final correctness test is that cached and uncached generation produce the
same logits, within floating-point tolerance.

## What belongs in GitHub

The repository should contain normal Python source files, tests, a tiny corpus,
and clear commands in the README. Generated checkpoints should normally be
ignored or published separately. Notebooks are optional presentation tools;
they should import the tested Python implementation rather than contain the
only copy of it.

The project is complete when I can explain the data flow, train the tiny model,
inspect its attention, and show why KV caching avoids repeated work—not when it
produces impressive prose.

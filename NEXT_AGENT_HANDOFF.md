# Handoff: keep the project as two small exercises

## User intent

The user wants exactly two related but independent learning artifacts:

1. a minimally trainable transformer made by stacking existing PyTorch layers;
2. a complete dummy transformer with fixed values whose forward pass can be
   calculated by hand.

Read `PROJECT_LEARNING_ROADMAP.md` before changing scope. Do not turn this into
a roadmap toward a miniature production language model.

## Current state

`demo/tiny_transformer.py` and its notebook implement a dependency-free,
single-head attention example with fixed embeddings, positions, and Q/K/V
matrices. Its intermediate values are inspectable, and its tests cover
position-dependent queries and keys plus causal masking.

It is only the beginning of the by-hand exercise. A complete forward pass still
needs:

- the attention output projection;
- the first residual connection and layer normalization;
- a small feed-forward network;
- the second residual connection and layer normalization;
- a vocabulary projection and final probabilities;
- a matching hand-calculated walkthrough and expected-value tests.

## Best implementation sequence

Finish the existing by-hand model first. Keep its dimensions and fixed numbers
as small as possible, even if that makes some matrices identities or zeros.
Every added operation must appear in the trace and in the walkthrough.

After that, add a separate `trainable/` implementation. Use PyTorch's existing
embedding, transformer, and linear layers with a causal mask. Its success
criterion is only that a tiny next-token training run lowers loss and that one
optimizer step changes a parameter.

## Guardrails

- Do not make the by-hand model trainable.
- Do not manually reimplement attention in the trainable model.
- Do not splice pretrained layers or download weights.
- Do not add GPU benchmarking, KV caching, multi-layer scaling, or production
  infrastructure.
- A notebook may explain calculations, but tested Python files remain the
  source of truth.
- Stop after each exercise meets the success criteria in the roadmap.

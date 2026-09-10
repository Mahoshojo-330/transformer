# Handoff: build an inspectable tiny transformer

## User goal

Help the user build a genuinely small transformer language model in a **new
repository**, while teaching every important idea as it is introduced. The
result should support light local training and make queries, keys, values,
attention weights, causal masking, and eventually the KV cache observable.

Do not continue implementation in the repository containing this note. Ask the
user for the new repository path, inspect that repository, and adapt the plan
to its existing state.

## User's current understanding

The user understands the basic attention flow:

```text
query compared with keys -> softmax weights -> weighted mixture of values
```

Important clarifications already established:

- `W_Q`, `W_K`, and `W_V` are model parameters shared across token positions.
- A query or key is not normally a permanent dictionary entry for a word. It
  is calculated for a particular token occurrence from its current hidden
  vector.
- At the first layer, that hidden vector includes token and position
  information. At later layers, it also includes context gathered by earlier
  layers.
- Without position information, swapping `river bank` to `bank river` simply
  swaps the first-layer token rows in unmasked self-attention.
- With position information, the same word at a different position generally
  receives a different query and key.
- A KV cache stores previous **key and value vectors**, not previous attention
  weights. Each new token creates a fresh query and a fresh attention row.

The user learns best from small numerical examples and wants to inspect the
calculations, not merely call a high-level pretrained model.

## Recommended scope

Build a tiny model from PyTorch primitives. Reuse open-source framework code,
but do not splice pretrained layers from unrelated models: their internal
representations are not generally compatible.

Keep version one deliberately constrained:

- word-level tokenizer with a tiny vocabulary;
- learned token and position embeddings;
- context length around 32;
- model dimension around 64;
- one or two transformer blocks;
- one to four causal-attention heads;
- residual connections, layer normalization, and a small feed-forward network;
- next-token cross-entropy training;
- CPU-friendly toy corpus;
- deterministic seed and small tests.

## Hardware-acceleration requirement

The user's machine is an Apple-silicon MacBook Pro with an Apple M5 Pro,
16-core integrated GPU, and 48 GB of unified memory. Use PyTorch's `mps` device
for GPU execution, with CPU as the reference and fallback backend. Do not add
CUDA-specific code to this project.

Treat device selection as configuration, not as a second implementation:

```python
if requested_device == "auto":
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
else:
    device = torch.device(requested_device)

model = model.to(device)
inputs = inputs.to(device)
targets = targets.to(device)
```

Expose `--device auto|cpu|mps` on training, generation, and inspection
commands. Log the selected device at startup. Verify `is_built()` and
`is_available()` in an environment-check command before training.

Acceleration milestones:

1. Make the forward pass and one optimizer step correct on CPU.
2. Run the same tests on MPS when available, comparing results with reasonable
   floating-point tolerances rather than exact equality.
3. Move the model and every batch tensor to one device; add assertions or tests
   that catch mixed-device mistakes.
4. Start with `float32`. Treat mixed precision as a later measured optimization,
   not a prerequisite.
5. Add a benchmark with warm-up steps and explicit MPS synchronization around
   timed regions.
6. Report step time and tokens per second for CPU and MPS.
7. Keep two configurations: a tiny readable teaching configuration and a
   larger benchmark configuration that creates enough matrix work to benefit
   from the GPU.

An extremely small transformer may be faster on CPU because GPU dispatch has
overhead. That is not a failure. First prove that MPS is used correctly, then
increase batch size, context length, or model width only in the benchmark until
the tradeoff becomes visible.

Do not silently enable CPU fallback for unsupported MPS operations at the
start; it can disguise performance problems. If fallback is genuinely needed,
document the reason and the `PYTORCH_ENABLE_MPS_FALLBACK` behavior. PyTorch's
official references are the [MPS backend
notes](https://docs.pytorch.org/docs/stable/notes/mps.html) and [MPS environment
variables](https://docs.pytorch.org/docs/stable/mps_environment_variables.html).

Use a tiny corpus containing both river and financial uses of `bank`. Be
explicit that such a corpus mainly demonstrates mechanics and overfitting; it
will not create broad language understanding.

## Build this incrementally

1. Establish the environment and a one-command test suite.
2. Implement the tokenizer and batches; show one encoded example.
3. Implement one attention head manually with PyTorch tensor operations. Return
   or capture `Q`, `K`, `V`, scores, masks, and weights for inspection.
4. Wrap it in one transformer block with residual paths, normalization, and an
   MLP. Explain the tensor shape at each boundary.
5. Add the vocabulary output head and verify a forward pass before training.
6. Add a short training script and prove that loss falls on the toy corpus.
7. Add device selection and prove that one training step works on CPU and MPS.
8. Add a synchronized CPU-versus-MPS benchmark.
9. Add generation with causal attention.
10. Add an inspection command for phrases such as `river bank` and `money bank`.
11. Add KV caching only after ordinary generation is correct; compare cached and
   uncached logits in a test.

At each stage, stop and explain the new concept with one concrete calculation.
Avoid scaffolding all future features before the current stage works.

## Suggested repository shape

```text
README.md
pyproject.toml                 # or requirements.txt; keep dependencies minimal
model.py                       # attention, block, tiny language model
tokenizer.py                   # small word-level tokenizer
train.py
generate.py
inspect_attention.py
data/tiny_corpus.txt
tests/test_model.py
tests/test_kv_cache.py         # add only with the KV-cache milestone
notebooks/                     # optional visualization only, not source of truth
checkpoints/                   # generated and normally gitignored
```

Ordinary Python modules and scripts should be the source of truth. A notebook
may later import them to visualize attention, but model or training logic
should not exist only in a notebook.

## Version-one success criteria

- Tests run with one documented command.
- A forward pass has documented tensor shapes.
- A short training run lowers loss on the supplied corpus.
- `--device cpu` and `--device mps` both complete a training step when MPS is
  available.
- A synchronized benchmark reports CPU and MPS throughput without claiming
  that the GPU must win for the smallest configuration.
- A generation command loads a saved checkpoint and emits tokens.
- An inspection command prints `Q`, `K`, `V`, attention scores, and weights for
  a chosen layer/head.
- Reversing two tokens visibly changes positional/contextual behavior.
- When KV caching is added, cached and uncached generation agree numerically.

## Environment note

In the old repository, Python was 3.14.7 and neither PyTorch nor NumPy was
installed. Creation of `.venv` began, but the PyTorch installation was
interrupted. Do not assume that environment is usable or carry it into the new
repository. Check the new environment and use a Python version supported by
the chosen current PyTorch release.

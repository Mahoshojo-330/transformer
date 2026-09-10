# Tiny transformer, by hand

This is a deliberately tiny implementation of **one self-attention head** using
only Python's standard library. It is a learning scaffold, not a production
model: there is no tokenizer, training, multi-head attention, MLP, layer
normalization, or output vocabulary yet.

Open the calculation walkthrough:

```sh
jupyter notebook demo.ipynb
```

The notebook compares `river bank` with `bank river`, both with and without
positional embeddings. Set `causal = True` in its configuration cell to see
decoder-style causal attention, where a token cannot inspect future tokens.

Run the tests:

```sh
python3 -m unittest -v
```

## Meaning versus position

For each token at position `i`, the model first constructs:

```text
x_i = word_embedding(token_i) + position_embedding(i)
```

It then calculates:

```text
q_i = x_i @ W_Q
k_i = x_i @ W_K
v_i = x_i @ W_V
```

`W_Q`, `W_K`, and `W_V` are fixed, hand-picked matrices in this project. In a
real transformer, training adjusts them. “Fixed matrices” does **not** mean
that every occurrence has a permanently fixed query and key: changing a
token's position changes `x_i`, which changes its query, key, and value.

Without positions, `river` has the same first-layer query and key in both
`river bank` and `bank river`. In unmasked attention, the entire result is
simply reordered. This is called permutation equivariance: attention sees
token content but not order. A causal mask additionally exposes which tokens
occur earlier, but it does not directly encode position or distance.

With positions, the two occurrences are different:

```text
river at position 0: [1, 0] + [0, 0]       = [1, 0]
river at position 1: [1, 0] + [0.25, -0.25] = [1.25, -0.25]
```

Because `W_Q` selects the second coordinate and `W_K` selects the first:

```text
river at position 0: q = [0],     k = [1]
river at position 1: q = [-0.25], k = [1.25]
```

That is how the example combines word identity and position. In deeper real
transformers, a token's input also contains context gathered by earlier
layers, so its query and key depend on meaning, position, **and context**.

## Reading the code in order

1. `teaching_model()` contains all numbers used by the model.
2. `FixedSelfAttention.forward()` creates `X`, `Q`, `K`, and `V`.
3. Each query is dot-multiplied with every key to produce `QK^T`.
4. `softmax()` turns each score row into attention weights.
5. Each output is the weighted sum of the value vectors.

Try changing the second position vector or swapping the two columns of an
embedding, then rerun the cells in `demo.ipynb` to see which calculations
change.

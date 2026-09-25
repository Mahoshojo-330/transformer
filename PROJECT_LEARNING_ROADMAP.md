# Roadmap: two small transformers

## The goal

Build two separate examples:

1. A tiny transformer that learns from a small piece of text.
2. A dummy transformer where every calculation can be done by hand.

They solve different problems, so they should stay separate.

## 1. Tiny model that trains

For this model, use transformer parts that PyTorch has already built. PyTorch
is open source, so we do not need to write attention or the other layers
ourselves.

The model will be:

```text
token numbers
  -> token and position embeddings
  -> one ready-made PyTorch transformer layer
  -> a layer that gives each possible next token a score
  -> calculate the error and train
```

The ready-made transformer layer is called `nn.TransformerEncoderLayer`. It
already contains attention, skip connections, normalization, and a small
feed-forward network.

We will reuse the layer code, but not mix together pretrained layers from
different models. All the layers will start with new weights and train
together. This avoids problems caused by unrelated pretrained layers expecting
different inputs.

### Steps

1. Turn a tiny piece of text into token numbers.
2. Connect the existing PyTorch layers.
3. Add a causal mask so a token cannot look at later tokens.
4. Train for a short time.
5. Show that the error, called the loss, becomes smaller.

### Finished when

- one command runs the training;
- the program prints the loss before and after training;
- the loss clearly gets smaller;
- a test shows that training changes the model's weights.

The model is allowed to memorize the tiny text. It does not need to produce
good writing.

## 2. Complete transformer calculated by hand

This model will not train. Its values will be picked by us so every step is
easy to follow with paper and a calculator.

It will use one transformer block, one attention head, and only two or three
tokens. Each vector will contain only two or three numbers.

The calculation will be:

```text
look up a small vector for each token and its position
  -> make its query, key, and value
  -> compare queries with keys
  -> hide future tokens
  -> turn the comparisons into attention weights
  -> mix the values using those weights
  -> add the original input and normalize it
  -> pass it through a small feed-forward network
  -> add the earlier result and normalize again
  -> calculate scores and probabilities for the next token
```

We will use small whole numbers, simple fractions, and identity or zero
matrices where helpful. The numbers are intentionally artificial because the
purpose is to understand the calculation.

### Steps

1. Finish the existing attention example in `demo/tiny_transformer.py`.
2. Add the missing skip connections, normalization, and feed-forward network.
3. Add the final layer that produces next-token probabilities.
4. Print every value produced along the way.
5. Write out one complete calculation in the same order as the code.

### Finished when

- every part of a transformer block is present;
- every intermediate value can be printed;
- one full result can be checked by hand;
- tests compare the code's values with the hand-calculated values.

## Build order

Finish the by-hand model first because its attention code already exists. Then
build the trainable model in a separate folder using PyTorch's ready-made
layers.

```text
demo/        # fixed numbers and hand calculations
trainable/   # ready-made PyTorch layers and minimal training
```

## Not part of this project

- mixing pretrained parts from unrelated models;
- making a useful language model;
- using many transformer blocks;
- GPU speed tests;
- KV caching;
- complex charts or training tools;
- production-quality data loading.

The project is finished when both small examples work and their purpose is
clear.

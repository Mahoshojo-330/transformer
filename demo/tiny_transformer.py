"""A tiny, dependency-free self-attention implementation for learning.

This intentionally implements one attention head, not a production transformer.
All intermediate values are returned so they can be checked by hand.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt
from typing import Sequence


Vector = list[float]
Matrix = list[Vector]


def dot(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("dot-product vectors must have the same length")
    return sum(a * b for a, b in zip(left, right))


def add(left: Sequence[float], right: Sequence[float]) -> Vector:
    if len(left) != len(right):
        raise ValueError("vectors must have the same length")
    return [a + b for a, b in zip(left, right)]


def transform(vector: Sequence[float], weights: Matrix) -> Vector:
    """Multiply a row vector by a weight matrix: vector @ weights."""
    if not weights or len(vector) != len(weights):
        raise ValueError("vector length must equal the number of matrix rows")
    width = len(weights[0])
    if any(len(row) != width for row in weights):
        raise ValueError("weight matrix rows must have equal lengths")
    return [sum(vector[i] * weights[i][j] for i in range(len(vector)))
            for j in range(width)]


def softmax(scores: Sequence[float]) -> Vector:
    """Convert scores to positive weights that sum to one."""
    finite_scores = [score for score in scores if score != float("-inf")]
    if not finite_scores:
        raise ValueError("softmax needs at least one unmasked score")
    maximum = max(finite_scores)
    numerators = [0.0 if score == float("-inf") else exp(score - maximum)
                  for score in scores]
    denominator = sum(numerators)
    return [number / denominator for number in numerators]


@dataclass(frozen=True)
class AttentionTrace:
    tokens: list[str]
    inputs: Matrix
    queries: Matrix
    keys: Matrix
    values: Matrix
    scores: Matrix
    weights: Matrix
    outputs: Matrix


class FixedSelfAttention:
    """One self-attention head whose embeddings and matrices are fixed.

    "Fixed" refers to the parameters. Queries, keys, and values are still
    calculated from each token's content-plus-position input.
    """

    def __init__(
        self,
        embeddings: dict[str, Vector],
        positions: Matrix,
        w_query: Matrix,
        w_key: Matrix,
        w_value: Matrix,
    ) -> None:
        self.embeddings = embeddings
        self.positions = positions
        self.w_query = w_query
        self.w_key = w_key
        self.w_value = w_value

    def forward(
        self,
        tokens: Sequence[str],
        *,
        use_positions: bool = True,
        causal: bool = False,
    ) -> AttentionTrace:
        if use_positions and len(tokens) > len(self.positions):
            raise ValueError("not enough position vectors for this sequence")

        inputs: Matrix = []
        for index, token in enumerate(tokens):
            if token not in self.embeddings:
                raise ValueError(f"unknown token: {token!r}")
            token_vector = self.embeddings[token]
            position = self.positions[index] if use_positions else [0.0] * len(token_vector)
            inputs.append(add(token_vector, position))

        queries = [transform(vector, self.w_query) for vector in inputs]
        keys = [transform(vector, self.w_key) for vector in inputs]
        values = [transform(vector, self.w_value) for vector in inputs]

        scale = sqrt(len(queries[0]))
        scores: Matrix = []
        weights: Matrix = []
        outputs: Matrix = []

        for query_index, query in enumerate(queries):
            row = [
                float("-inf") if causal and key_index > query_index
                else dot(query, key) / scale
                for key_index, key in enumerate(keys)
            ]
            attention_weights = softmax(row)
            output = [
                sum(attention_weights[j] * values[j][dimension]
                    for j in range(len(values)))
                for dimension in range(len(values[0]))
            ]
            scores.append(row)
            weights.append(attention_weights)
            outputs.append(output)

        return AttentionTrace(
            tokens=list(tokens),
            inputs=inputs,
            queries=queries,
            keys=keys,
            values=values,
            scores=scores,
            weights=weights,
            outputs=outputs,
        )


def teaching_model() -> FixedSelfAttention:
    """Return the two-word model used in the accompanying walkthrough."""
    return FixedSelfAttention(
        embeddings={
            "river": [1.0, 0.0],
            "bank": [0.0, 1.0],
        },
        # Easy-to-calculate stand-ins for learned positional embeddings.
        positions=[
            [0.0, 0.0],
            [0.25, -0.25],
        ],
        # [x1, x2] @ W_Q = [x2]
        w_query=[[0.0], [1.0]],
        # [x1, x2] @ W_K = [x1]
        w_key=[[1.0], [0.0]],
        # Values retain both input coordinates.
        w_value=[[1.0, 0.0], [0.0, 1.0]],
    )

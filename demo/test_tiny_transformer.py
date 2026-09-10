import unittest

from tiny_transformer import teaching_model


class FixedSelfAttentionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = teaching_model()

    def test_attention_rows_sum_to_one(self) -> None:
        trace = self.model.forward(["river", "bank"])
        for row in trace.weights:
            self.assertAlmostEqual(sum(row), 1.0)

    def test_without_positions_same_word_has_same_query_and_key(self) -> None:
        first = self.model.forward(["river", "bank"], use_positions=False)
        second = self.model.forward(["bank", "river"], use_positions=False)
        self.assertEqual(first.queries[0], second.queries[1])
        self.assertEqual(first.keys[0], second.keys[1])

    def test_with_positions_same_word_changes_query_and_key(self) -> None:
        first = self.model.forward(["river", "bank"], use_positions=True)
        second = self.model.forward(["bank", "river"], use_positions=True)
        self.assertNotEqual(first.queries[0], second.queries[1])
        self.assertNotEqual(first.keys[0], second.keys[1])

    def test_causal_attention_cannot_look_forward(self) -> None:
        trace = self.model.forward(["river", "bank"], causal=True)
        self.assertEqual(trace.weights[0][1], 0.0)


if __name__ == "__main__":
    unittest.main()

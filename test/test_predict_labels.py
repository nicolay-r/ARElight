import unittest

from arelight.predict.header import PredictHeader
from arelight.predict.provider import IntPredictProvider
from arelight.run.infer import attitude_task_kwargs


class TestPredictLabels(unittest.TestCase):

    def test_uint_selects_named_column(self):
        uint_labels = [1, 2, 0]
        rows = list(IntPredictProvider.handle_iter(
            data_it=[(7, 1), (8, 2), (9, 0)],
            uint_labels=uint_labels))
        self.assertEqual(PredictHeader.create_header(uint_labels=uint_labels),
                         ["id", "col_1", "col_2", "col_0"])
        self.assertEqual(rows[0], [7, "1", "0", "0"])
        self.assertEqual(rows[1], [8, "0", "1", "0"])
        self.assertEqual(rows[2], [9, "0", "0", "1"])

    def test_raw_out_is_cast_to_uint(self):
        rows = list(IntPredictProvider.handle_iter(
            data_it=[(7, " 1 "), (8, '"2"'), (9, "'0'")],
            uint_labels=[1, 2, 0]))
        self.assertEqual(rows[0], [7, "1", "0", "0"])
        self.assertEqual(rows[1], [8, "0", "1", "0"])
        self.assertEqual(rows[2], [9, "0", "0", "1"])

    def test_attitude_prompt_asks_for_label_id(self):
        task = attitude_task_kwargs("u:0,p:1,n:2")
        self.assertNotIn("classify_func", task)
        prompt = task["schema"][0]["prompt"]
        self.assertIn("{text_a}", prompt)
        self.assertIn("0 for neutral, 1 for positive, 2 for negative", prompt)
        self.assertEqual(task["schema"][0]["out"], "response")


if __name__ == "__main__":
    unittest.main()

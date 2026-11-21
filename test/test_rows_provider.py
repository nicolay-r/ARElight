import os
import tempfile
import unittest
from os.path import dirname, join, realpath

from arelight.api import _rows_as_dict, create_inference_pipeline
from arelight.run.infer import iter_rows_from_files
from arelight.doc_provider import InMemoryRowsDocProvider


DATA_DIR = join(dirname(realpath(__file__)), "data")


class TestInMemoryRows(unittest.TestCase):

    def _provider(self, rows, docs_limit=None):
        return InMemoryRowsDocProvider(
            rows=rows,
            content_to_sentences=lambda text: [text],
            docs_limit=docs_limit)

    def test_mapping(self):
        provider = self._provider({"left": "one side", "right": {"text": "other side"}})
        self.assertEqual(provider.by_id("right").get_sentence(0).Text, "other side")
        self.assertEqual(list(provider.iter_doc_ids()), ["left", "right"])

    def test_text_key_and_limit(self):
        provider = self._provider(
            {"doc-a": {"text": "first"}, "doc-b": {"text": "second"}, "doc-c": "third"},
            docs_limit=2)
        self.assertEqual(list(provider.iter_doc_ids()), ["doc-a", "doc-b"])
        keyed = InMemoryRowsDocProvider(
            rows={"doc-b": {"body": "second"}},
            content_to_sentences=lambda text: [text],
            text_key="body")
        self.assertEqual(keyed.by_id("doc-b").get_sentence(0).Text, "second")

    def test_rows_require_id(self):
        rows = _rows_as_dict([{"id": "doc-a", "text": "first"}])
        self.assertEqual(rows["doc-a"]["text"], "first")
        with self.assertRaises(KeyError):
            _rows_as_dict([{"text": "missing"}])

    def test_list_is_rejected(self):
        with self.assertRaises(AssertionError):
            self._provider(["alpha text", "beta text"])


class TestFileRows(unittest.TestCase):

    def test_text_file(self):
        path = join(DATA_DIR, "rus_input_text_example.txt")
        rows = list(iter_rows_from_files([path]))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], path + ":0")
        self.assertIn("Франция", rows[0]["text"])

    def test_csv_column(self):
        path = join(DATA_DIR, "responses-d3js-backend-sample-data.csv")
        rows = list(iter_rows_from_files([path], csv_column="text_a"))
        self.assertGreater(len(rows), 1)
        self.assertTrue(rows[0]["text"])


class _FakeNER(object):
    pass


class TestPipelineFromRows(unittest.TestCase):

    def test_pipeline_has_no_graph_backend(self):
        with tempfile.TemporaryDirectory() as folder:
            llm_path = join(folder, "fake_llm.py")
            with open(llm_path, "w") as handle:
                handle.write("class FakeLLM:\n"
                             "    def __init__(self, api_token=None, model_name=None, **kwargs):\n"
                             "        self.api_token = api_token\n")

            pipeline, settings = create_inference_pipeline(
                rows_iter={"sample:0": "Barack Obama met Angela Merkel in Berlin."},
                sampling_args={
                    "sentence_parser": "nltk:english",
                    "terms_per_context": 50,
                    "docs_limit": 1,
                },
                predict_table_name="bulk_chain",
                collection_target_func=lambda data_type: join(folder, data_type.name.lower()),
                translator_args=None,
                ner_args={"model": _FakeNER(), "obj_filter": None, "chunk_limit": 128},
                inference_args={
                    "model": "fake",
                    "class_name": llm_path,
                    "api_key": "x",
                    "writer": "sqlite3",
                    "task_kwargs": {"labels_fmt": "u:0,p:1,n:2"},
                },
            )

        self.assertFalse(any(type(item).__name__.startswith("D3js") for item in pipeline))
        merged = {}
        for chunk in settings:
            merged.update(chunk)
        self.assertNotIn("d3js_collection_name", merged)
        self.assertNotIn("labels_formatter", merged)
        self.assertTrue(merged["predict_filepath"].endswith(".sqlite"))


if __name__ == "__main__":
    unittest.main()

import asyncio
import importlib.util
import unittest

STREAM_PATH = "/home/nicolay/proj/united/arelight/stream_arelight.py"


def _load_stream():
    spec = importlib.util.spec_from_file_location("stream_arelight", STREAM_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestStreamUsesRows(unittest.TestCase):

    def test_posted_text_is_a_row_and_graphs_are_requested(self):
        stream_mod = _load_stream()
        captured = {}

        class _Provider(object):
            def __init__(self, *args, **kwargs):
                pass

        def fake_create(**kwargs):
            captured.update(kwargs)
            return [], [{}]

        stream_mod.dynamic_init = lambda class_filepath: _Provider
        stream_mod.create_inference_pipeline = fake_create
        stream_mod.BasePipelineLauncher.run = lambda **kwargs: None

        async def collect():
            chunks = []
            async for chunk in stream_mod.stream_answer("Obama met Merkel in Berlin."):
                chunks.append(chunk)
            return chunks

        chunks = asyncio.get_event_loop().run_until_complete(collect())

        rows = list(captured["rows_iter"])
        self.assertEqual(rows, [{"id": "request:0", "text": "Obama met Merkel in Berlin."}])
        self.assertEqual(captured["backend_args"]["collection_name"], "test_samples")
        self.assertEqual(captured["backend_args"]["output_dir"], "./output")
        self.assertIn("event: completed", chunks[0])
        self.assertNotIn("files_iter", captured)


if __name__ == "__main__":
    unittest.main()

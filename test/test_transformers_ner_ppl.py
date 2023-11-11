import unittest

from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.pipelines.items.entities_ner_transformers import TransformersNERPipelineItem
from arelight.pipelines.items.terms_splitter import CustomTermsSplitterPipelineItem
from arelight.utils import IdAssigner
from utils_ner import test_ner


class TestTransformersNERPipeline(unittest.TestCase):

    def get_texts(self):
        with open("data/book_sample.txt", "r") as f:
            text = f.read()
        return [text]

    def test_transformers(self):
        # Declare input texts.

        ppl_items = [
            TransformersNERPipelineItem(id_assigner=IdAssigner(), ner_model_name="dslim/bert-base-NER"),
            CustomTermsSplitterPipelineItem(),
        ]

        test_ner(texts=self.get_texts(),
                 ner_ppl_items=ppl_items,
                 prefix="transformers-ner")

    def test_benchmark(self):

        ppl_items = [
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(id_assigner=IdAssigner(), ner_model_name="ner_ontonotes_bert")
        ]

        test_ner(texts=self.get_texts(),
                 ner_ppl_items=ppl_items,
                 prefix="dp-ner")

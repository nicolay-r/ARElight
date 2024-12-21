import unittest
import time
from tqdm import tqdm

from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.pipelines.items.entities_ner_transformers import TransformersNERPipelineItem
from arelight.third_party.transformers import annotate_ner_ppl, init_token_classification_model, annotate_ner
from arelight.utils import IdAssigner
from utils_ner import test_ner


class TestTransformersNERPipeline(unittest.TestCase):

    def get_texts(self):
        with open("data/book_sample.txt", "r") as f:
            text = f.read()
        return [text]

    def test_transformers_batch(self):
        sentences = self.get_texts()[0].split("\n")
        model, tokenizer = init_token_classification_model(model_path="dslim/bert-base-NER", device="cpu")

        print("Sentences: {}".format(len(sentences)))

        # E1.
        start = time.time()
        for s in tqdm(sentences):
            annotate_ner(model=model, tokenizer=tokenizer, text=s)
        end = time.time()
        print(end - start)

        # E2.
        batch_size = 8
        start = time.time()
        ppl = annotate_ner_ppl(model=model, tokenizer=tokenizer, batch_size=batch_size)
        for i in tqdm(range(0, len(sentences), batch_size)):
            b = sentences[i:i + batch_size]
            if len(b) != batch_size:
                b += [""] * (batch_size - len(b))
            ppl(b)
        end = time.time()
        print(end - start)

    def test_transformers(self):
        # Declare input texts.

        ppl_items = [
            TransformersNERPipelineItem(id_assigner=IdAssigner(),
                                        ner_model_name="dslim/bert-base-NER", device="cpu",
                                        src_func=lambda s: s.Text)
        ]

        test_ner(texts=self.get_texts(), ner_ppl_items=ppl_items, collection_name="transformers-ner")

    def test_benchmark(self):

        ppl_items = [
            BasePipelineItem(src_func=lambda s: s.Text),
            DeepPavlovNERPipelineItem(id_assigner=IdAssigner(),
                                      src_func=lambda text: split_by_whitespaces(text),
                                      ner_model_name="ner_ontonotes_bert")
        ]

        test_ner(texts=self.get_texts(),
                 ner_ppl_items=ppl_items,
                 collection_name="dp-ner")

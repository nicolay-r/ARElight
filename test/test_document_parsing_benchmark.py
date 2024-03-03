from arekit.common.data.input.providers.const import IDLE_MODE
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.enums import TermFormat
from arekit.common.text.parsed import BaseParsedText
from arekit.common.utils import split_by_whitespaces

import utils
import unittest

from arekit.common.docs.parser import DocumentParsers
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.utils.pipelines.items.text.translator import MLTextTranslatorPipelineItem

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.pipelines.items.entities_ner_transformers import TransformersNERPipelineItem
from arelight.run.utils import create_translate_model
from arelight.utils import IdAssigner


class DocumentParsingBenchmark(unittest.TestCase):
    """ This unit test represent a tutorial on how
        AREkit might be applied towards data preparation for BERT
        model.
    """

    @staticmethod
    def read_text(filepath):
        with open(filepath, "r") as f:
            return f.read()

    def test_ner_deeppavlov(self):

        # Declare text parser.
        text_parser_pipeline = [
            BasePipelineItem(src_func=lambda s: s.Text),
            DeepPavlovNERPipelineItem(
                src_func=lambda text: split_by_whitespaces(text),
                id_assigner=IdAssigner(),
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_name="ner_ontonotes_bert_mult"),
        ]

        # Composing labels formatter and experiment preparation.
        text = DocumentParsingBenchmark.read_text("data/book-war-and-peace-test.txt")
        doc_provider = utils.InMemoryDocProvider(docs=utils.input_to_docs([text]))

        print("Sentences:", doc_provider.by_id(0).SentencesCount)
        pd = DocumentParsers.parse_batch(doc=doc_provider.by_id(0),
                                         pipeline_items=text_parser_pipeline,
                                         parent_ppl_ctx=PipelineContext(d={IDLE_MODE: None}),
                                         batch_size=16,
                                         show_progress=True)

        for s in pd:
            assert (isinstance(s, BaseParsedText))
            for t in s.iter_terms(TermFormat.Raw):
                print(t)

    def test_ner_transformers(self):

        # Declare text parser.
        text_parser_pipeline = [
            BasePipelineItem(src_func=lambda s: s.Text),
            TransformersNERPipelineItem(id_assigner=IdAssigner(),
                                        ner_model_name="dslim/bert-base-NER", device="cpu"),
        ]

        # Composing labels formatter and experiment preparation.
        text = DocumentParsingBenchmark.read_text("data/book-war-and-peace-test.txt")
        doc_provider = utils.InMemoryDocProvider(docs=utils.input_to_docs([text]))

        print("Sentences:", doc_provider.by_id(0).SentencesCount)
        pd = DocumentParsers.parse_batch(doc=doc_provider.by_id(0),
                                         pipeline_items=text_parser_pipeline,
                                         parent_ppl_ctx=PipelineContext(d={IDLE_MODE: None}),
                                         batch_size=16,
                                         show_progress=True)

        for s in pd:
            assert(isinstance(s, BaseParsedText))
            for t in s.iter_terms(TermFormat.Raw):
                print(t)

    def test_translator(self):

        translator = create_translate_model("googletrans")

        # Declare text parser.
        text_parser_pipeline = [
            BasePipelineItem(src_func=lambda s: s.Text),
            MLTextTranslatorPipelineItem(
                src_func=lambda text: split_by_whitespaces(text),
                batch_translate_model=lambda content: translator(str_list=content, src="ru", dest="en"),
                do_translate_entity=False)
        ]

        # Composing labels formatter and experiment preparation.
        text = DocumentParsingBenchmark.read_text("data/book-war-and-peace-test-ru.txt")
        doc_provider = utils.InMemoryDocProvider(docs=utils.input_to_docs([text]))

        print("Sentences:", doc_provider.by_id(0).SentencesCount)

        pd = DocumentParsers.parse_batch(doc=doc_provider.by_id(0),
                                         pipeline_items=text_parser_pipeline,
                                         parent_ppl_ctx=PipelineContext(d={IDLE_MODE: None}),
                                         batch_size=16,
                                         show_progress=True)

        for s in pd:
            assert(isinstance(s, BaseParsedText))
            for t in s.iter_terms(TermFormat.Raw):
                print(t)


if __name__ == '__main__':
    unittest.main()

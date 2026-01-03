import utils
import unittest

from bulk_ner.src.pipeline.item.ner import NERPipelineItem
from bulk_ner.src.utils import IdAssigner

from arekit.common.data.input.providers.const import IDLE_MODE
from arekit.common.pipeline.context import PipelineContext
from arekit.common.text.enums import TermFormat
from arekit.common.text.parsed import BaseParsedText
from arekit.common.utils import split_by_whitespaces
from arekit.common.docs.parser import DocumentParsers
from arekit.common.pipeline.items.base import BasePipelineItem

from bulk_translate.src.pipeline.translator import MLTextTranslatorPipelineItem

from arelight.arekit.indexed_entity import IndexedEntity
from arelight.utils import get_event_loop
from core.dp_130 import DeepPavlovNER
from gt_402 import GoogleTranslateModel


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
            NERPipelineItem(id_assigner=IdAssigner(),
                            src_func=lambda text: split_by_whitespaces(text),
                            model=DeepPavlovNER(model="ner_ontonotes_bert_mult", download=False, install=False),
                            obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                            # It is important to provide the correct type (see AREkit #575)
                            create_entity_func=lambda value, e_type, entity_id: IndexedEntity(value=value, e_type=e_type, entity_id=entity_id),
                            chunk_limit=128),
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
                print(t, )

    def test_translator(self):

        translator = GoogleTranslateModel()

        event_loop = get_event_loop()

        # Declare text parser.
        text_parser_pipeline = [
            BasePipelineItem(src_func=lambda s: s.Text),
            MLTextTranslatorPipelineItem(
                src_func=lambda text: split_by_whitespaces(text),
                batch_translate_model=translator.get_func(src="ru", dest="en", event_loop=event_loop),
                is_span_func=lambda term: isinstance(term, IndexedEntity),
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
                print(t, )


if __name__ == '__main__':
    unittest.main()

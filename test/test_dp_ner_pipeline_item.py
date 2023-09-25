import unittest
import utils
from os.path import join

from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.entities.base import Entity
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.enums import TermFormat
from arekit.common.text.parsed import BaseParsedText
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.pipelines.items.entity import IndexedEntity
from arelight.run.utils import read_synonyms_collection
from arelight.utils import IdAssigner


class BertOntonotesPipelineItemTest(unittest.TestCase):
    """ Support text chunking.
    """

    def test_pipeline_item_rus(self):

        # Declaring text processing pipeline.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(
                id_assigner=IdAssigner(),
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_name="ner_ontonotes_bert_mult"),
        ])

        # Read file contents.
        text_filepath = join(utils.TEST_DATA_DIR, "rus_input_text_example.txt")
        with open(text_filepath, 'r') as f:
            text = f.read().rstrip()

        # Output parsed text.
        parsed_text = text_parser.run(text)
        for t in parsed_text.iter_terms(TermFormat.Raw):
            print("<{}> ({})".format(t.Value, t.Type) if isinstance(t, Entity) else t)

    def test_pipeline(self):

        text = ".. При этом Москва неоднократно подчеркивала, что ее активность " \
               "на балтике является ответом именно на действия НАТО и эскалацию " \
               "враждебного подхода к Росcии вблизи ее восточных границ ..."

        synonyms = read_synonyms_collection(join(utils.TEST_DATA_DIR, "rus_synonyms_rusentrel.txt"))

        # Declare text parser.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(id_assigner=IdAssigner(), ner_model_name="ner_ontonotes_bert_mult"),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ])

        # Launch pipeline.
        parsed_text = text_parser.run(text)
        assert(isinstance(parsed_text, BaseParsedText))

        for term in parsed_text.iter_terms(TermFormat.Raw):
            if isinstance(term, IndexedEntity):
                print(term.ID, term.GroupIndex, term.Value)
            else:
                print(term)

    def test_pipeline_item_eng_book(self):

        # Declaring text processing pipeline.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(
                id_assigner=IdAssigner(),
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_name="ner_ontonotes_bert"),
        ])

        # Read file contents.
        text_filepath = join(utils.TEST_DATA_DIR, "book-war-and-peace-test.txt")
        with open(text_filepath, 'r') as f:
            text = f.read().rstrip()

        # Output parsed text.
        parsed_text = text_parser.run(text)
        for t in parsed_text.iter_terms(TermFormat.Raw):
            print("<{}> ({})".format(t.Value, t.Type) if isinstance(t, Entity) else t)


if __name__ == '__main__':
    unittest.main()

import utils
from os.path import join

import unittest

from arekit.common.docs.base import Document
from arekit.common.docs.parser import DocumentParsers
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.entities.base import Entity
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.enums import TermFormat
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.pipelines.items.entity import IndexedEntity
from arelight.run.utils import iter_group_values
from arelight.utils import IdAssigner


class BertOntonotesPipelineItemTest(unittest.TestCase):
    """ Support text chunking.
    """

    def test_pipeline_item_rus(self):

        # Declaring text processing pipeline.
        pipeline_items = [
            DeepPavlovNERPipelineItem(
                src_func=lambda text: split_by_whitespaces(text),
                id_assigner=IdAssigner(),
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_name="ner_ontonotes_bert_mult"),
        ]

        # Read file contents.
        text_filepath = join(utils.TEST_DATA_DIR, "rus_input_text_example.txt")
        with open(text_filepath, 'r') as f:
            text = f.read().rstrip()

        # Output parsed text.
        parsed_doc = DocumentParsers.parse_batch(doc=Document(doc_id=0, sentences=[text]), pipeline_items=pipeline_items, batch_size=16)
        for t in parsed_doc.get_sentence(0).iter_terms(TermFormat.Raw):
            print("<{}> ({})".format(t.Value, t.Type) if isinstance(t, Entity) else t)

    def test_pipeline(self):

        text = ".. При этом Москва неоднократно подчеркивала, что ее активность " \
               "на балтике является ответом именно на действия НАТО и эскалацию " \
               "враждебного подхода к Росcии вблизи ее восточных границ ..."

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=iter_group_values(join(utils.TEST_DATA_DIR, "rus_synonyms_rusentrel.txt")),
            stemmer=MystemWrapper(), is_read_only=False)

        # Declare text parser.
        pipeline_items = [
            DeepPavlovNERPipelineItem(
                src_func=lambda t: split_by_whitespaces(t),
                id_assigner=IdAssigner(),
                ner_model_name="ner_ontonotes_bert_mult"),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ]

        # Launch pipeline.
        parsed_doc = DocumentParsers.parse_batch(doc=Document(doc_id=0, sentences=[text]), pipeline_items=pipeline_items, batch_size=16)
        for term in parsed_doc.get_sentence(0).iter_terms(TermFormat.Raw):
            if isinstance(term, IndexedEntity):
                print(term.ID, term.GroupIndex, term.Value)
            else:
                print(term)

    def test_pipeline_item_eng_book(self):

        # Declaring text processing pipeline.
        pipeline_items = [
            DeepPavlovNERPipelineItem(
                src_func=lambda t: split_by_whitespaces(t),
                id_assigner=IdAssigner(),
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_name="ner_ontonotes_bert"),
        ]

        # Read file contents.
        text_filepath = join(utils.TEST_DATA_DIR, "book-war-and-peace-test.txt")
        with open(text_filepath, 'r') as f:
            text = f.read().rstrip()

        # Output parsed text.
        parsed_doc = DocumentParsers.parse_batch(doc=Document(doc_id=0, sentences=[text]), pipeline_items=pipeline_items, batch_size=16)
        for t in parsed_doc.get_sentence(0).iter_terms(TermFormat.Raw):
            print("<{}> ({})".format(t.Value, t.Type) if isinstance(t, Entity) else t)


if __name__ == '__main__':
    unittest.main()

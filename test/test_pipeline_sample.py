import utils
import unittest

from os.path import join

from arekit.common.utils import split_by_whitespaces
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.data import const
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.contrib.utils.data.writers.sqlite_native import SQliteWriter
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage

from arelight.arekit.samples_io import CustomSamplesIO
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.items.entities_default import TextEntitiesParser
from arelight.pipelines.items.serializer_arekit import AREkitSerializerPipelineItem
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes
from arelight.synonyms import iter_synonym_groups
from arelight.utils import IdAssigner


class EntityFilter(object):

    def __init__(self):
        pass

    def is_ignored(self, entity, e_type):
        raise NotImplementedError()


class BertTestSerialization(unittest.TestCase):
    """ This unit test represent a tutorial on how
        AREkit might be applied towards data preparation for BERT
        model.
    """

    @staticmethod
    def iter_groups(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for data in iter_synonym_groups(file):
                yield data

    def test(self):

        # Declare input texts.
        texts = [
            # Text 1.
            """24 марта президент [США] [Джо-Байден] провел переговоры с
               лидерами стран [Евросоюза] в [Брюсселе] , вызвав внимание рынка и предположения о
               том, что [Америке] удалось уговорить [ЕС] совместно бойкотировать российские нефть
               и газ.  [Европейский-Союз] крайне зависим от [России] в плане поставок нефти и
               газа."""
        ]

        # Declare synonyms collection.
        synonyms_filepath = join(utils.TEST_DATA_DIR, "rus_synonyms_rusentrel.txt")

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=self.iter_groups(synonyms_filepath),
            stemmer=MystemWrapper(),
            is_read_only=False)

        # Declare text parser.
        text_parser_pipeline = [
            BasePipelineItem(src_func=lambda s: s.Text),
            TextEntitiesParser(src_func=lambda s: split_by_whitespaces(s), id_assigner=IdAssigner()),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ]

        # Composing labels formatter and experiment preparation.
        doc_provider = utils.InMemoryDocProvider(docs=utils.input_to_docs(texts))
        pipeline = [
            AREkitSerializerPipelineItem(
                rows_provider=create_bert_sample_provider(
                    label_scaler=SingleLabelScaler(NoLabel()),
                    provider_type=BertSampleProviderTypes.NLI_M,
                    entity_formatter=SharpPrefixedEntitiesSimpleFormatter(),
                    crop_window=50),
                save_labels_func=lambda _: False,
                samples_io=CustomSamplesIO(
                    create_target_func=lambda data_type: join(utils.TEST_OUT_DIR, "samples", data_type.name.lower()),
                    writer=SQliteWriter()),
                storage=RowCacheStorage(force_collect_columns=[
                    const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND
            ]))
        ]
        synonyms = StemmerBasedSynonymCollection(iter_group_values_lists=[],
                                                 stemmer=MystemWrapper(),
                                                 is_read_only=False)

        # Initialize data processing pipeline.
        test_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                           dist_in_terms_bound=50,
                                                           dist_in_sentences=0,
                                                           doc_provider=doc_provider,
                                                           text_pipeline=text_parser_pipeline,
                                                           terms_per_context=50,
                                                           batch_size=10)

        BasePipelineLauncher.run(pipeline=pipeline,
                                 pipeline_ctx=PipelineContext(d={
                                     "doc_ids": list(range(len(texts))),
                                     "data_type_pipelines": {DataType.Test: test_pipeline}
                                 }),
                                 src_key="doc_ids")


if __name__ == '__main__':
    unittest.main()

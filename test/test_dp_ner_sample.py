from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem

import utils
import unittest
from rusenttokenize import ru_sent_tokenize

from arekit.common.docs.base import Document
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.docs.sentence import BaseDocumentSentence
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.base import BasePipeline
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.synonyms.utils import iter_synonym_groups
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection

from arelight.doc_provider import InMemoryDocProvider
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem

from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes
from arelight.utils import IdAssigner


class TestCombinedPipeline(unittest.TestCase):
    """ This unit test represent a tutorial on how
        AREkit might be applied towards data preparation for BERT
        model.
    """

    @staticmethod
    def input_to_docs(texts):
        assert(isinstance(texts, list))
        docs = []
        for doc_id, contents in enumerate(texts):
            sentences = ru_sent_tokenize(contents)
            sentences = list(map(lambda text: BaseDocumentSentence(text), sentences))
            doc = Document(doc_id=doc_id, sentences=sentences)
            docs.append(doc)
        return docs

    @staticmethod
    def iter_groups(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for data in iter_synonym_groups(file):
                yield data

    def test(self):

        # Declare input texts.
        texts = [
            # Text 1.
            """24 марта президент США Джо-Байден провел переговоры с
               лидерами стран Евросоюза в Брюсселе , вызвав внимание рынка и предположения о
               том, что Америке удалось уговорить ЕС совместно бойкотировать российские нефть
               и газ.  Европейский-Союз крайне зависим от России в плане поставок нефти и
               газа."""
        ]

        synonyms = SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

        # Declare text parser.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(id_assigner=IdAssigner(), ner_model_name="ner_ontonotes_bert_mult"),
            EntitiesGroupingPipelineItem(lambda value:
                                         SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                                             synonyms=synonyms, value=value))
        ])

        # Single label scaler.
        single_label_scaler = SingleLabelScaler(NoLabel())

        # Composing labels formatter and experiment preparation.
        doc_provider = InMemoryDocProvider(docs=self.input_to_docs(texts))

        rows_provider = create_bert_sample_provider(
            label_scaler=single_label_scaler,
            provider_type=BertSampleProviderTypes.NLI_M,
            entity_formatter=SharpPrefixedEntitiesSimpleFormatter())

        pipeline = BasePipeline([
            BaseSerializerPipelineItem(
                rows_provider=rows_provider,
                storage=RowCacheStorage(),
                samples_io=SamplesIO(target_dir=utils.TEST_OUT_DIR, writer=NativeCsvWriter(delimiter=',')),
                save_labels_func=lambda data_type: data_type != DataType.Test)
        ])

        # Initialize data processing pipeline.
        test_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                           dist_in_terms_bound=50,
                                                           dist_in_sentences=0,
                                                           doc_provider=doc_provider,
                                                           text_parser=text_parser,
                                                           terms_per_context=50)

        pipeline.run(input_data=PipelineContext({
                         "doc_ids": list(range(len(texts))),
                         "data_type_pipelines": {DataType.Test: test_pipeline}
                     }))


if __name__ == '__main__':
    unittest.main()

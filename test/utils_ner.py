from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.base import BasePipeline
from arekit.common.pipeline.context import PipelineContext
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection

from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes

import utils


def test_ner(texts, ner_ppl_items, prefix):
    assert(isinstance(texts, list))

    synonyms = SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

    # Declare text parser.
    text_parser = BaseTextParser(pipeline=ner_ppl_items + [
        EntitiesGroupingPipelineItem(
            lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))
    ])

    # Single label scaler.
    single_label_scaler = SingleLabelScaler(NoLabel())

    # Composing labels formatter and experiment preparation.
    doc_provider = utils.InMemoryDocProvider(docs=utils.input_to_docs(texts))

    rows_provider = create_bert_sample_provider(
        label_scaler=single_label_scaler,
        provider_type=BertSampleProviderTypes.NLI_M,
        entity_formatter=SharpPrefixedEntitiesSimpleFormatter(),
        crop_window=50)

    pipeline = BasePipeline([
        BaseSerializerPipelineItem(
            rows_provider=rows_provider,
            storage=RowCacheStorage(),
            samples_io=SamplesIO(target_dir=utils.TEST_OUT_DIR,
                                 writer=NativeCsvWriter(delimiter=','),
                                 prefix=prefix),
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

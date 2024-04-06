from os.path import join

from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.context import PipelineContext
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection

from arelight.arekit.samples_io import CustomSamplesIO
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes

import utils


def test_ner(texts, ner_ppl_items, collection_name):
    assert(isinstance(texts, list))

    synonyms = SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

    # Declare text parser.
    text_pipeline_items = ner_ppl_items + [
        EntitiesGroupingPipelineItem(
            lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))
    ]

    # Single label scaler.
    single_label_scaler = SingleLabelScaler(NoLabel())

    # Composing labels formatter and experiment preparation.
    doc_provider = utils.InMemoryDocProvider(docs=utils.input_to_docs(texts))

    rows_provider = create_bert_sample_provider(
        label_scaler=single_label_scaler,
        provider_type=BertSampleProviderTypes.NLI_M,
        entity_formatter=SharpPrefixedEntitiesSimpleFormatter(),
        crop_window=50)

    # Target function.
    create_target_func = lambda data_type: join(
        utils.TEST_OUT_DIR, "-".join([collection_name, data_type.name.lower()]))

    pipeline_items = [
        BaseSerializerPipelineItem(
            rows_provider=rows_provider,
            storage=RowCacheStorage(log_out=None),
            samples_io=CustomSamplesIO(create_target_func=create_target_func, writer=NativeCsvWriter(delimiter=',')),
            save_labels_func=lambda data_type: data_type != DataType.Test)
    ]

    # Initialize data processing pipeline.
    test_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                       dist_in_terms_bound=50,
                                                       dist_in_sentences=0,
                                                       doc_provider=doc_provider,
                                                       text_pipeline=text_pipeline_items,
                                                       terms_per_context=50,
                                                       batch_size=10)

    ctx = PipelineContext({
        "doc_ids": list(range(len(texts))),
        "data_type_pipelines": {DataType.Test: test_pipeline}
    })

    BasePipelineLauncher.run(pipeline=pipeline_items, pipeline_ctx=ctx, has_input=False)

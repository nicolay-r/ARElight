import argparse
from os.path import join, dirname, basename

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.pipeline.base import BasePipeline
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.doc_ops import InMemoryDocOperations
from arelight.pipelines.annot_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.items.utils import input_to_docs
from arelight.run.args import common
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser)
    common.EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    common.FromDataframeArg.add_argument(parser)
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.PredictOutputFilepathArg.add_argument(parser, default=const.OUTPUT_TEMPLATE)
    common.BertTextBFormatTypeArg.add_argument(parser, default='nli_m')
    common.SentenceParserArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Parsing arguments.
    text_from_arg = common.InputTextArg.read_argument(args)
    texts_from_files = common.FromFilesArg.read_argument(args)
    texts_from_dataframe = common.FromDataframeArg.read_argument(args)
    entities_parser = common.EntitiesParserArg.read_argument(args)
    sentence_parser = common.SentenceParserArg.read_argument(args)
    entity_fmt = create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args))
    input_texts = text_from_arg if text_from_arg is not None else \
        texts_from_files if texts_from_files is not None else texts_from_dataframe
    opin_annot = BaseOpinionAnnotator()
    doc_ops = InMemoryDocOperations(docs=input_to_docs(input_texts, sentence_parser=sentence_parser))
    labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})
    label_scaler = SingleLabelScaler(NoLabel())
    backend_template = common.PredictOutputFilepathArg.read_argument(args)

    synonyms = read_synonyms_collection(
        filepath=common.SynonymsCollectionFilepathArg.read_argument(args))

    annot_algo = PairBasedOpinionAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_provider=ConstantLabelProvider(label_instance=NoLabel()))

    # Declare text parser.
    text_parser = BaseTextParser(pipeline=[
        TermsSplitterParser(),
        entities_parser,
        EntitiesGroupingPipelineItem(lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value))
    ])

    terms_per_context = common.TermsPerContextArg.read_argument(args)

    # Initialize data processing pipeline.
    test_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                       dist_in_terms_bound=terms_per_context,
                                                       terms_per_context=terms_per_context,
                                                       doc_ops=doc_ops,
                                                       text_parser=text_parser,
                                                       dist_in_sentences=0)

    rows_provider = create_bert_sample_provider(
        label_scaler=label_scaler,
        provider_type=BertSampleProviderTypes.NLI_M,
        entity_formatter=entity_fmt)

    pipeline = BasePipeline([
        BertExperimentInputSerializerPipelineItem(
            rows_provider=rows_provider,
            storage=RowCacheStorage(),
            samples_io=SamplesIO(target_dir=dirname(backend_template),
                                 prefix=basename(backend_template),
                                 writer=NativeCsvWriter(delimiter=',')),
            save_labels_func=lambda data_type: data_type != DataType.Test,
            balance_func=lambda data_type: data_type == DataType.Train)
    ])

    no_folding = NoFolding(doc_ids=list(range(len(texts_from_files))), supported_data_type=DataType.Test)

    pipeline.run(input_data=None,
                 params_dict={
                     "data_folding": no_folding,
                     "data_type_pipelines": {DataType.Test: test_pipeline}
                 })

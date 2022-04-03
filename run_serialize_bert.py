import argparse

from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter

from network.args import const
from network.args.common import InputTextArg, EntitiesParserArg, RusVectoresEmbeddingFilepathArg, \
    TermsPerContextArg, SynonymsCollectionArg, FromFileArg, EntityFormatterTypesArg
from network.args.const import TEXT_DEFAULT
from pipelines.serialize_bert import BertTextSerializationPipelineItem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    InputTextArg.add_argument(parser, default=None)
    FromFileArg.add_argument(parser, default=TEXT_DEFAULT)
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    SynonymsCollectionArg.add_argument(parser, default=None)

    # Parsing arguments.
    args = parser.parse_args()

    ppl = BasePipeline([
        BertTextSerializationPipelineItem(
            terms_per_context=TermsPerContextArg.read_argument(args),
            synonyms=SynonymsCollectionArg.read_argument(args),
            entities_parser=EntitiesParserArg.read_argument(args),
            name_provider=ExperimentNameProvider(name="example-bert", suffix="serialize"),
            entity_fmt=create_entity_formatter(EntityFormatterTypesArg.read_argument(args)),
            opin_annot=DefaultAnnotator(annot_algo=PairBasedAnnotationAlgorithm(
                dist_in_terms_bound=None,
                label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            data_folding=NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test]))
    ])

    text_from_arg = InputTextArg.read_argument(args)
    text_from_file = FromFileArg.read_argument(args)

    ppl.run(text_from_arg if text_from_arg is not None else text_from_file)
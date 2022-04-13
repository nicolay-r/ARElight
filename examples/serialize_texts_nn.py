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

from examples.args import const
from examples.args import common
from examples.args.const import DEFAULT_TEXT_FILEPATH

from arelight.pipelines.serialize_nn import NetworkTextsSerializationPipelineItem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[DEFAULT_TEXT_FILEPATH])
    common.EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    common.RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-simple-eng")
    common.StemmerArg.add_argument(parser, default="mystem")
    common.SynonymsCollectionArg.add_argument(parser, default=None)
    common.FramesColectionArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    ppl = BasePipeline([
        NetworkTextsSerializationPipelineItem(
            terms_per_context=common.TermsPerContextArg.read_argument(args),
            synonyms=common.SynonymsCollectionArg.read_argument(args),
            entities_parser=common.EntitiesParserArg.read_argument(args),
            embedding_path=common.RusVectoresEmbeddingFilepathArg.read_argument(args),
            name_provider=ExperimentNameProvider(name="example", suffix="serialize"),
            entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
            opin_annot=DefaultAnnotator(annot_algo=PairBasedAnnotationAlgorithm(
                dist_in_terms_bound=None,
                label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            stemmer=common.StemmerArg.read_argument(args),
            frames_collection=common.FramesColectionArg.read_argument(args),
            data_folding=NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test]))
    ])

    text_from_arg = common.InputTextArg.read_argument(args)
    text_from_file = common.FromFilesArg.read_argument(args)

    ppl.run(text_from_arg if text_from_arg is not None else text_from_file)

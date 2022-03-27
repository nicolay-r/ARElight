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

from input import EXAMPLES
from network.args import const
from network.args.common import InputTextArg, EntitiesParserArg, RusVectoresEmbeddingFilepathArg, TermsPerContextArg, \
    StemmerArg, SynonymsCollectionArg, FramesColectionArg
from network.args.serialize import EntityFormatterTypesArg
from pipelines.serialize import TextSerializationPipelineItem

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    InputTextArg.add_argument(parser, default=EXAMPLES["no_entities"][0])
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    EntityFormatterTypesArg.add_argument(parser, default="hidden-simple-eng")
    StemmerArg.add_argument(parser, default="mystem")
    SynonymsCollectionArg.add_argument(parser, default=None)
    FramesColectionArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    ppl = BasePipeline([
        TextSerializationPipelineItem(
            terms_per_context=TermsPerContextArg.read_argument(args),
            synonyms=SynonymsCollectionArg.read_argument(args),
            entities_parser=EntitiesParserArg.read_argument(args),
            embedding_path=RusVectoresEmbeddingFilepathArg.read_argument(args),
            name_provider=ExperimentNameProvider(name="example", suffix="serialize"),
            entity_fmt=create_entity_formatter(EntityFormatterTypesArg.read_argument(args)),
            opin_annot=DefaultAnnotator(annot_algo=PairBasedAnnotationAlgorithm(
                dist_in_terms_bound=None,
                label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            stemmer=StemmerArg.read_argument(args),
            frames_collection=FramesColectionArg.read_argument(args),
            data_folding=NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test]))
    ])

    ppl.run(InputTextArg.read_argument(args))

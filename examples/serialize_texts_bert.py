import argparse
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.base import BaseOpinionAnnotator
from arekit.common.pipeline.base import BasePipeline
from arelight.pipelines.serialize_bert import BertTextsSerializationPipelineItem

from examples.args import const, common
from examples.args.const import DEFAULT_TEXT_FILEPATH
from examples.utils import read_synonyms_collection

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[DEFAULT_TEXT_FILEPATH])
    common.EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.BertTextBFormatTypeArg.add_argument(parser, default='nli_m')

    # Parsing arguments.
    args = parser.parse_args()

    text_from_arg = common.InputTextArg.read_argument(args)
    texts_from_files = common.FromFilesArg.read_argument(args)

    synonyms_collection = read_synonyms_collection(
        filepath=common.SynonymsCollectionFilepathArg.read_argument(args))

    ppl = BasePipeline([
        BertTextsSerializationPipelineItem(
            terms_per_context=common.TermsPerContextArg.read_argument(args),
            synonyms=synonyms_collection,
            output_dir=const.OUTPUT_DIR,
            entities_parser=common.EntitiesParserArg.read_argument(args),
            name_provider=ExperimentNameProvider(name="example-bert", suffix="serialize"),
            entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
            text_b_type=common.BertTextBFormatTypeArg.read_argument(args),
            opin_annot=BaseOpinionAnnotator(annot_algo=PairBasedOpinionAnnotationAlgorithm(
                dist_in_terms_bound=None,
                label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            data_folding=NoFolding(doc_ids_to_fold=list(range(len(texts_from_files))),
                                   supported_data_types=[DataType.Test]))
    ])

    ppl.run(text_from_arg if text_from_arg is not None else texts_from_files)

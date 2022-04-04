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
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNegativeLabel, ExperimentPositiveLabel
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter

from network.args import const
from network.args.common import LabelsCountArg, InputTextArg, FromFileArg, SynonymsCollectionArg, \
    EntityFormatterTypesArg, EntitiesParserArg, TermsPerContextArg, PredictOutputFilepathArg, BertConfigFilepathArg, \
    BertCheckpointFilepathArg, BertVocabFilepathArg
from network.args.const import BERT_VOCAB_PATH, BERT_CKPT_PATH, BERT_CONFIG_PATH
from pipelines.backend import BratBackendPipelineItem
from pipelines.inference_bert import BertInferencePipelineItem
from pipelines.serialize_bert import BertTextSerializationPipelineItem
from utils import create_labels_scaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    InputTextArg.add_argument(parser, default=None)
    FromFileArg.add_argument(parser, default=const.TEXT_DEFAULT)
    SynonymsCollectionArg.add_argument(parser, default=None)
    LabelsCountArg.add_argument(parser, default=3)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    PredictOutputFilepathArg.add_argument(parser, default=None)
    BertConfigFilepathArg.add_argument(parser, default=BERT_CONFIG_PATH)
    BertCheckpointFilepathArg.add_argument(parser, default=BERT_CKPT_PATH)
    BertVocabFilepathArg.add_argument(parser, default=BERT_VOCAB_PATH)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading text-related parameters.
    text_from_file = FromFileArg.read_argument(args)
    text_from_arg = InputTextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else text_from_file

    # Implement extra structures.
    labels_scaler = create_labels_scaler(LabelsCountArg.read_argument(args))

    # Parsing arguments.
    args = parser.parse_args()

    # Declaring pipeline.
    ppl = BasePipeline(pipeline=[

        BertTextSerializationPipelineItem(
            synonyms=SynonymsCollectionArg.read_argument(args),
            terms_per_context=TermsPerContextArg.read_argument(args),
            entities_parser=EntitiesParserArg.read_argument(args),
            entity_fmt=create_entity_formatter(EntityFormatterTypesArg.read_argument(args)),
            name_provider=ExperimentNameProvider(name="example-bert", suffix="infer"),
            opin_annot=DefaultAnnotator(
                PairBasedAnnotationAlgorithm(
                    dist_in_terms_bound=None,
                    label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            data_folding=NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])),

        BertInferencePipelineItem(
            predict_writer=TsvPredictWriter(),
            bert_config_file=BertConfigFilepathArg.read_argument(args),
            model_checkpoint_path=BertCheckpointFilepathArg.read_argument(args),
            vocab_filepath=BertVocabFilepathArg.read_argument(args),
            data_type=DataType.Test,
            labels_scaler=labels_scaler),

        BratBackendPipelineItem(label_to_rel={
                str(labels_scaler.label_to_uint(ExperimentPositiveLabel())): "POS",
                str(labels_scaler.label_to_uint(ExperimentNegativeLabel())): "NEG"
            },
            obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN", "NEG": "RED"},
        )
    ])

    backend_template = PredictOutputFilepathArg.read_argument(args)

    ppl.run(actual_content, {
        "predict_fp": "{}.npz".format(backend_template) if backend_template is not None else None,
        "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None
    })

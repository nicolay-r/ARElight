import argparse
from os.path import join

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

from arelight.pipelines.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.inference_bert import BertInferencePipelineItem
from arelight.pipelines.serialize_bert import BertTextsSerializationPipelineItem

from examples.args import common
from examples.args import train
from examples.args import const
from examples.args.train import DoLowercaseArg
from examples.utils import create_labels_scaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[const.DEFAULT_TEXT_FILEPATH])
    common.SynonymsCollectionArg.add_argument(parser, default=None)
    common.LabelsCountArg.add_argument(parser, default=3)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.TokensPerContextArg.add_argument(parser, default=128)
    common.EntitiesParserArg.add_argument(parser, default=const.DEFAULT_ENTITIES_PARSER)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    common.PredictOutputFilepathArg.add_argument(parser, default=None)
    common.BertCheckpointFilepathArg.add_argument(parser, default=const.BERT_FINETUNED_CKPT_PATH)
    common.BertConfigFilepathArg.add_argument(parser, default=const.BERT_CONFIG_PATH)
    common.BertVocabFilepathArg.add_argument(parser, default=const.BERT_VOCAB_PATH)
    common.BertTextBFormatTypeArg.add_argument(parser, default='nli_m')
    train.DoLowercaseArg.add_argument(parser, default=const.BERT_DO_LOWERCASE)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading text-related parameters.
    texts_from_files = common.FromFilesArg.read_argument(args)
    text_from_arg = common.InputTextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else texts_from_files

    # Implement extra structures.
    labels_scaler = create_labels_scaler(common.LabelsCountArg.read_argument(args))

    # Parsing arguments.
    args = parser.parse_args()

    # Declaring pipeline.
    ppl = BasePipeline(pipeline=[

        BertTextsSerializationPipelineItem(
            synonyms=common.SynonymsCollectionArg.read_argument(args),
            terms_per_context=common.TermsPerContextArg.read_argument(args),
            entities_parser=common.EntitiesParserArg.read_argument(args),
            entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
            name_provider=ExperimentNameProvider(name="example-bert", suffix="infer"),
            text_b_type=common.BertTextBFormatTypeArg.read_argument(args),
            output_dir=const.OUTPUT_DIR,
            opin_annot=DefaultAnnotator(
                PairBasedAnnotationAlgorithm(
                    dist_in_terms_bound=None,
                    label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            data_folding=NoFolding(doc_ids_to_fold=list(range(len(texts_from_files))),
                                   supported_data_types=[DataType.Test])),

        BertInferencePipelineItem(
            data_type=DataType.Test,
            predict_writer=TsvPredictWriter(),
            bert_config_file=common.BertConfigFilepathArg.read_argument(args),
            model_checkpoint_path=common.BertCheckpointFilepathArg.read_argument(args),
            vocab_filepath=common.BertVocabFilepathArg.read_argument(args),
            max_seq_length=common.TokensPerContextArg.read_argument(args),
            do_lowercase=DoLowercaseArg.read_argument(args),
            labels_scaler=labels_scaler),

        BratBackendContentsPipelineItem(label_to_rel={
                str(labels_scaler.label_to_uint(ExperimentPositiveLabel())): "POS",
                str(labels_scaler.label_to_uint(ExperimentNegativeLabel())): "NEG"
            },
            obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN", "NEG": "RED"},
        ),

        BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/")
    ])

    backend_template = common.PredictOutputFilepathArg.read_argument(args)

    ppl.run(actual_content, {
        "template_filepath": join(const.DATA_DIR, "brat_template.html"),
        "predict_fp": "{}.npz".format(backend_template) if backend_template is not None else None,
        "brat_vis_fp": "{}.html".format(backend_template) if backend_template is not None else None
    })

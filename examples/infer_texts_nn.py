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
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames

from arelight.pipelines.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.inference_nn import TensorflowNetworkInferencePipelineItem
from arelight.pipelines.serialize_nn import NetworkTextsSerializationPipelineItem
from arelight.network.nn.common import create_full_model_name, create_network_model_io, create_bags_collection_type

from examples.args import const, common, train
from examples.utils import create_labels_scaler, read_synonyms_collection

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[const.DEFAULT_TEXT_FILEPATH])
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    common.LabelsCountArg.add_argument(parser, default=3)
    common.ModelNameArg.add_argument(parser, default=ModelNames.PCNN.value)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-simple-eng")
    common.VocabFilepathArg.add_argument(parser, default=None)
    common.EmbeddingMatrixFilepathArg.add_argument(parser, default=None)
    common.ModelLoadDirArg.add_argument(parser, default=const.NEURAL_NETWORKS_TARGET_DIR)
    common.EntitiesParserArg.add_argument(parser, default=const.DEFAULT_ENTITIES_PARSER)
    common.StemmerArg.add_argument(parser, default="mystem")
    common.PredictOutputFilepathArg.add_argument(parser, default=None)
    common.FramesColectionArg.add_argument(parser)
    train.BagsPerMinibatchArg.add_argument(parser, default=const.BAGS_PER_MINIBATCH)
    train.ModelInputTypeArg.add_argument(parser, default=ModelInputType.SingleInstance)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    model_name = common.ModelNameArg.read_argument(args)
    model_input_type = train.ModelInputTypeArg.read_argument(args)
    model_load_dir = common.ModelLoadDirArg.read_argument(args)
    frames_collection = common.FramesColectionArg.read_argument(args)

    # Reading text-related parameters.
    texts_from_files = common.FromFilesArg.read_argument(args)
    text_from_arg = common.InputTextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else texts_from_files

    # Implement extra structures.
    labels_scaler = create_labels_scaler(common.LabelsCountArg.read_argument(args))

    # Parsing arguments.
    args = parser.parse_args()

    #############################
    # Execute pipeline element.
    #############################
    full_model_name = create_full_model_name(model_name=model_name, input_type=model_input_type)

    nn_io = create_network_model_io(
        full_model_name=full_model_name,
        embedding_filepath=common.EmbeddingMatrixFilepathArg.read_argument(args),
        source_dir=model_load_dir,
        target_dir=model_load_dir,
        vocab_filepath=common.VocabFilepathArg.read_argument(args),
        model_name_tag=u'')

    synonyms_collection = read_synonyms_collection(
        filepath=common.SynonymsCollectionFilepathArg.read_argument(args))

    # Declaring pipeline.
    ppl = BasePipeline(pipeline=[

        NetworkTextsSerializationPipelineItem(
            frames_collection=frames_collection,
            synonyms=synonyms_collection,
            terms_per_context=common.TermsPerContextArg.read_argument(args),
            embedding_path=common.RusVectoresEmbeddingFilepathArg.read_argument(args),
            entities_parser=common.EntitiesParserArg.read_argument(args),
            entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
            stemmer=common.StemmerArg.read_argument(args),
            name_provider=ExperimentNameProvider(name="example", suffix="infer"),
            opin_annot=DefaultAnnotator(
                PairBasedAnnotationAlgorithm(
                    dist_in_terms_bound=None,
                    label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            output_dir=const.OUTPUT_DIR,
            data_folding=NoFolding(doc_ids_to_fold=list(range(len(texts_from_files))),
                                   supported_data_types=[DataType.Test])),

        TensorflowNetworkInferencePipelineItem(
            nn_io=nn_io,
            model_name=model_name,
            data_type=DataType.Test,
            bags_per_minibatch=train.BagsPerMinibatchArg.read_argument(args),
            bags_collection_type=create_bags_collection_type(model_input_type=model_input_type),
            model_input_type=model_input_type,
            labels_scaler=labels_scaler,
            predict_writer=TsvPredictWriter(),
            callbacks=[
                TrainingLimiterCallback(train_acc_limit=0.99),
                TrainingStatProviderCallback(),
            ]),

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

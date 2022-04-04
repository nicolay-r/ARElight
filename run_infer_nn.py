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
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames

from network.args import const
from network.args.common import ModelNameArg, LabelsCountArg, RusVectoresEmbeddingFilepathArg, SynonymsCollectionArg, \
    InputTextArg, TermsPerContextArg, VocabFilepathArg, EmbeddingMatrixFilepathArg, ModelLoadDirArg, EntitiesParserArg, \
    StemmerArg, PredictOutputFilepathArg, FramesColectionArg, FromFilesArg, EntityFormatterTypesArg
from network.args.train import ModelInputTypeArg, BagsPerMinibatchArg
from network.nn.common import create_network_model_io, create_bags_collection_type, create_full_model_name
from pipelines.backend import BratBackendPipelineItem
from pipelines.inference_nn import TensorflowNetworkInferencePipelineItem
from pipelines.serialize_nn import NetworkTextsSerializationPipelineItem
from utils import create_labels_scaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    InputTextArg.add_argument(parser, default=None)
    FromFilesArg.add_argument(parser, default=[const.DEFAULT_TEXT_FILEPATH])
    SynonymsCollectionArg.add_argument(parser, default=None)
    RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    BagsPerMinibatchArg.add_argument(parser, default=const.BAGS_PER_MINIBATCH)
    LabelsCountArg.add_argument(parser, default=3)
    ModelNameArg.add_argument(parser, default=ModelNames.PCNN.value)
    ModelInputTypeArg.add_argument(parser, default=ModelInputType.SingleInstance)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    EntityFormatterTypesArg.add_argument(parser, default="hidden-simple-eng")
    VocabFilepathArg.add_argument(parser, default=None)
    EmbeddingMatrixFilepathArg.add_argument(parser, default=None)
    ModelLoadDirArg.add_argument(parser, default=const.NEURAL_NETWORKS_TARGET_DIR)
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    StemmerArg.add_argument(parser, default="mystem")
    PredictOutputFilepathArg.add_argument(parser, default=None)
    FramesColectionArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading provided arguments.
    model_name = ModelNameArg.read_argument(args)
    model_input_type = ModelInputTypeArg.read_argument(args)
    model_load_dir = ModelLoadDirArg.read_argument(args)
    frames_collection = FramesColectionArg.read_argument(args)

    # Reading text-related parameters.
    texts_from_files = FromFilesArg.read_argument(args)
    text_from_arg = InputTextArg.read_argument(args)
    actual_content = text_from_arg if text_from_arg is not None else texts_from_files

    # Implement extra structures.
    labels_scaler = create_labels_scaler(LabelsCountArg.read_argument(args))

    # Parsing arguments.
    args = parser.parse_args()

    #############################
    # Execute pipeline element.
    #############################
    full_model_name = create_full_model_name(model_name=model_name, input_type=model_input_type)

    nn_io = create_network_model_io(
        full_model_name=full_model_name,
        embedding_filepath=EmbeddingMatrixFilepathArg.read_argument(args),
        source_dir=model_load_dir,
        target_dir=model_load_dir,
        vocab_filepath=VocabFilepathArg.read_argument(args),
        model_name_tag=u'')

    # Declaring pipeline.
    ppl = BasePipeline(pipeline=[
        NetworkTextsSerializationPipelineItem(
            frames_collection=frames_collection,
            synonyms=SynonymsCollectionArg.read_argument(args),
            terms_per_context=TermsPerContextArg.read_argument(args),
            embedding_path=RusVectoresEmbeddingFilepathArg.read_argument(args),
            entities_parser=EntitiesParserArg.read_argument(args),
            entity_fmt=create_entity_formatter(EntityFormatterTypesArg.read_argument(args)),
            stemmer=StemmerArg.read_argument(args),
            name_provider=ExperimentNameProvider(name="example", suffix="infer"),
            opin_annot=DefaultAnnotator(
                PairBasedAnnotationAlgorithm(
                    dist_in_terms_bound=None,
                    label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            data_folding=NoFolding(doc_ids_to_fold=list(range(len(texts_from_files))),
                                   supported_data_types=[DataType.Test])),
        TensorflowNetworkInferencePipelineItem(
            nn_io=nn_io,
            model_name=model_name,
            data_type=DataType.Test,
            bags_per_minibatch=BagsPerMinibatchArg.read_argument(args),
            bags_collection_type=create_bags_collection_type(model_input_type=model_input_type),
            model_input_type=model_input_type,
            labels_scaler=labels_scaler,
            predict_writer=TsvPredictWriter(),
            callbacks=[
                TrainingLimiterCallback(train_acc_limit=0.99),
                TrainingStatProviderCallback(),
            ]),
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

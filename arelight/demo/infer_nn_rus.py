from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from arekit.contrib.experiment_rusentrel.entities.types import EntityFormattersService
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.processing.lemmatization.mystem import MystemWrapper

from arelight.demo.utils import read_synonyms_collection
from arelight.network.nn.common import create_network_model_io, create_bags_collection_type, create_full_model_name
from arelight.pipelines.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.inference_nn import TensorflowNetworkInferencePipelineItem
from arelight.pipelines.serialize_nn import NetworkTextsSerializationPipelineItem
from arelight.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem


def demo_infer_texts_tensorflow_nn_pipeline(texts_count,
                                            model_name, model_input_type, model_load_dir,
                                            frames_collection,
                                            output_dir,
                                            synonyms_filepath,
                                            embedding_filepath,
                                            embedding_matrix_filepath=None,
                                            vocab_filepath=None,
                                            bags_per_minibatch=2,
                                            entity_fmt_type=EntityFormattersService.name_to_type("hidden-simple-eng"),
                                            exp_name_provider=ExperimentNameProvider(name="example", suffix="infer"),
                                            stemmer=MystemWrapper(),
                                            labels_scaler=ThreeLabelScaler(),
                                            terms_per_context=50):
    assert(isinstance(texts_count, int))
    assert(isinstance(model_name, ModelNames))

    nn_io = create_network_model_io(
        full_model_name=create_full_model_name(model_name=model_name, input_type=model_input_type),
        embedding_filepath=embedding_matrix_filepath,
        source_dir=model_load_dir,
        target_dir=model_load_dir,
        vocab_filepath=vocab_filepath,
        model_name_tag=u'')

    # Declaring pipeline.
    ppl = BasePipeline(pipeline=[

        NetworkTextsSerializationPipelineItem(
            frames_collection=frames_collection,
            synonyms=read_synonyms_collection(synonyms_filepath=synonyms_filepath, stemmer=stemmer),
            terms_per_context=terms_per_context,
            embedding_path=embedding_filepath,
            entities_parser=BertOntonotesNERPipelineItem(
                lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"]),
            entity_fmt=create_entity_formatter(entity_fmt_type),
            stemmer=stemmer,
            name_provider=exp_name_provider,
            opin_annot=DefaultAnnotator(
                PairBasedAnnotationAlgorithm(
                    dist_in_terms_bound=None,
                    label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            output_dir=output_dir,
            data_folding=NoFolding(doc_ids_to_fold=list(range(texts_count)),
                                   supported_data_types=[DataType.Test])),

        TensorflowNetworkInferencePipelineItem(
            nn_io=nn_io,
            model_name=model_name,
            data_type=DataType.Test,
            bags_per_minibatch=bags_per_minibatch,
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
    ])

    return ppl

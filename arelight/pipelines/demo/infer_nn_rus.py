from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.input.term_types import TermTypes
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015
from arekit.contrib.utils.vectorizers.bpe import BPEVectorizer
from arekit.contrib.utils.vectorizers.random_norm import RandomNormalVectorizer

from arelight.network.nn.common import create_bags_collection_type, create_full_model_name
from arelight.network.nn.ctx import CustomNeuralNetworkSerializationContext
from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler
from arelight.pipelines.items.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.items.inference_nn import TensorflowNetworkInferencePipelineItem


def demo_infer_texts_tensorflow_nn_pipeline(texts_count,
                                            model_name,
                                            model_input_type,
                                            model_load_dir,
                                            output_dir,
                                            entity_fmt,
                                            frames_collection,
                                            bags_per_minibatch=2,
                                            labels_scaler=ThreeLabelScaler()):
    assert(isinstance(texts_count, int))
    assert(isinstance(model_name, ModelNames))

    nn_io = NeuralNetworkModelIO(
        full_model_name=create_full_model_name(model_name=model_name, input_type=model_input_type),
        source_dir=model_load_dir,
        target_dir=model_load_dir,
        model_name_tag=u'')

    PairBasedOpinionAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_provider=ConstantLabelProvider(label_instance=NoLabel()))

    stemmer = MystemWrapper()
    embedding = load_embedding_news_mystem_skipgram_1000_20_2015(stemmer)
    bpe_vectorizer = BPEVectorizer(embedding=embedding, max_part_size=3)
    norm_vectorizer = RandomNormalVectorizer(vector_size=embedding.VectorSize,
                                             token_offset=12345)

    ctx = CustomNeuralNetworkSerializationContext(
        labels_scaler=labels_scaler,
        pos_tagger=POSMystemWrapper(stemmer.MystemInstance),
        frames_collection=frames_collection)

    samples_io = SamplesIO(target_dir=output_dir)
    emb_io = NpEmbeddingIO(target_dir=output_dir)

    # Declaring pipeline.
    pipeline = BasePipeline(pipeline=[

        NetworksInputSerializerPipelineItem(
            vectorizers={
                TermTypes.WORD: bpe_vectorizer,
                TermTypes.ENTITY: bpe_vectorizer,
                TermTypes.FRAME: bpe_vectorizer,
                TermTypes.TOKEN: norm_vectorizer
            },
            ctx=ctx,
            str_entity_fmt=entity_fmt,
            samples_io=samples_io,
            emb_io=emb_io,
            save_labels_func=lambda data_type: data_type != DataType.Test,
            balance_func=lambda data_type: data_type == DataType.Train,
            save_embedding=True),

        TensorflowNetworkInferencePipelineItem(
            nn_io=nn_io,
            emb_io=emb_io,
            samples_io=samples_io,
            model_name=model_name,
            data_type=DataType.Test,
            bag_size=1,
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
            str(labels_scaler.label_to_uint(PositiveLabel())): "POS",
            str(labels_scaler.label_to_uint(NegativeLabel())): "NEG"
        },
            obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN", "NEG": "RED"},
        ),
    ])

    return pipeline

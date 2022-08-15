from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.embedding import Embedding
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.processing.pos.base import POSTagger


class CustomNeuralNetworkSerializationContext(NetworkSerializationContext):

    def __init__(self, labels_scaler, pos_tagger, embedding, terms_per_context, str_entity_formatter,
                 frames_collection, frame_variant_collection):
        assert(isinstance(embedding, Embedding))
        assert(isinstance(pos_tagger, POSTagger))
        assert(isinstance(frames_collection, RuSentiFramesCollection))
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))

        super(NetworkSerializationContext, self).__init__(label_scaler=labels_scaler)

        self.__pos_tagger = pos_tagger
        self.__terms_per_context = terms_per_context
        self.__str_entity_formatter = str_entity_formatter
        self.__word_embedding = embedding
        self.__frames_collection = frames_collection
        self.__frame_variant_collection = frame_variant_collection
        self.__frame_roles_label_scaler = None # ThreeLabelScaler()
        self.__frames_connotation_provider = RuSentiFramesConnotationProvider(collection=self.__frames_collection)

    @property
    def PosTagger(self):
        return self.__pos_tagger

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

    @property
    def FramesConnotationProvider(self):
        return self.__frames_connotation_provider

    @property
    def FrameRolesLabelScaler(self):
        return self.__frame_roles_label_scaler

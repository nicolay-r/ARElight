import importlib
import logging
from enum import Enum

from arekit.contrib.source.synonyms.utils import iter_synonym_groups
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler
from arelight.pipelines.items.entities_default import TextEntitiesParser
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_entity_parser(ner_model_name, id_assigner, obj_filter_types=None):
    """ NOTE: It is important that the IdAssigner is expected to be unique for all the
        entity parsers.

        obj_filter_types: str
    """
    assert(isinstance(ner_model_name, str) or ner_model_name is None)
    assert(isinstance(obj_filter_types, list) or obj_filter_types is None)

    if ner_model_name is None:
        return TextEntitiesParser(id_assigner)
    else:
        return DeepPavlovNERPipelineItem(
            obj_filter=None if obj_filter_types is None else lambda s_obj: s_obj.ObjectType in obj_filter_types,
            ner_model_name=ner_model_name,
            id_assigner=id_assigner)


def create_sentence_parser(arg):
    if arg == "linesplit":
        return lambda text: [t.strip() for t in text.split('\n')]
    elif arg == "ru":
        # Using ru_sent_tokenize library.
        ru_sent_tokenize = importlib.import_module("ru_sent_tokenize")
        return lambda text: ru_sent_tokenize.ru_sent_tokenize(text)
    elif arg == "nltk_en":
        # Using nltk library.
        nltk = importlib.import_module("nltk")
        tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
        return tokenizer.tokenize
    else:
        raise Exception("Arg `{}` was not found".format(arg))


def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")


def read_synonyms_collection(filepath):

    def __iter_groups(filepath):
        with open(filepath, 'r') as file:
            for group in iter_synonym_groups(file):
                yield group

    return StemmerBasedSynonymCollection(
        iter_group_values_lists=__iter_groups(filepath),
        stemmer=MystemWrapper(),
        is_read_only=False)


class EnumConversionService(object):

    _data = None

    @classmethod
    def is_supported(cls, name):
        assert(isinstance(cls._data, dict))
        return name in cls._data

    @classmethod
    def name_to_type(cls, name):
        assert(isinstance(cls._data, dict))
        assert(isinstance(name, str))
        return cls._data[name]

    @classmethod
    def iter_names(cls):
        assert(isinstance(cls._data, dict))
        return iter(list(cls._data.keys()))

    @classmethod
    def type_to_name(cls, enum_type):
        assert(isinstance(cls._data, dict))
        assert(isinstance(enum_type, Enum))

        for item_name, item_type in cls._data.items():
            if item_type == enum_type:
                return item_name

        raise NotImplemented("Formatting type '{}' does not supported".format(enum_type))


def merge_dictionaries(dict_iter):
    d_out = {}
    for d in dict_iter:
        for k, v in d.items():
            if k in d_out:
                raise Exception("Key `{}` is already registred!".format(k))
            d_out[k] = v
    return d_out

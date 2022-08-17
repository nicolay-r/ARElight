from enum import Enum

from arekit.contrib.source.synonyms.utils import iter_synonym_groups
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from arelight.demo.labels.scalers import ThreeLabelScaler


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
        is_read_only=False,
        debug=False)


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

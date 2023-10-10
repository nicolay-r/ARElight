import importlib
import time
import logging
from enum import Enum

from arekit.contrib.source.synonyms.utils import iter_synonym_groups
from googletrans import Translator

from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SingletonTranslator(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Translator()
        return cls._instance


def translate_value(value, src, dest, attempts=10):
    translator = SingletonTranslator.instance()

    import logging
    logger = logging.getLogger()  # get the default logger
    logger.setLevel(50)

    for i in range(attempts):
        try:
            return translator.translate(value, dest=dest, src=src).text
        except:
            logger.info("Unable to perform translation. Try {} out of {}.".format(i, attempts))
            time.sleep(1)

    raise Exception("Can't translate")


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


def iter_group_values(filepath):

    if filepath is None:
        return None

    with open(filepath, 'r') as file:
        for group in iter_synonym_groups(file):
            yield group


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
    merged_dict = {}
    for d in dict_iter:
        for key, value in d.items():
            if key in merged_dict:
                raise Exception("Key `{}` is already registred!".format(key))
            merged_dict[key] = value
    return merged_dict


def read_files(paths):

    if paths is None:
        return None

    file_contents = []
    for path in paths:
        with open(path) as f:
            file_contents.append(f.read().rstrip())

    return file_contents

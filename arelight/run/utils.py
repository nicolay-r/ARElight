import importlib
import logging
from enum import Enum

from arekit.common.docs.base import Document
from arekit.common.docs.sentence import BaseDocumentSentence
from arekit.contrib.source.synonyms.utils import iter_synonym_groups

from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.utils import auto_import, iter_csv_lines

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_sentence_parser(arg):
    if arg == "linesplit":
        return lambda text: [t.strip() for t in text.split('\n')]
    elif arg == "ru":
        # Using ru_sent_tokenize library.
        ru_sent_tokenize = importlib.import_module("ru_sent_tokenize")
        return lambda text: ru_sent_tokenize.ru_sent_tokenize(text)
    elif arg == "nltk_en":
        # Using nltk library.
        tokenizer_func = auto_import("arelight.third_party.nltk.import_tokenizer")
        tokenizer = tokenizer_func(name='tokenizers/punkt/english.pickle', resource_name="punkt")
        return tokenizer.tokenize
    else:
        raise Exception("Arg `{}` was not found".format(arg))


def iter_content(filepath, csv_column, csv_delimiter):
    if filepath.endswith(".csv"):
        for line in iter_csv_lines(filepath, column_name=csv_column, delimiter=csv_delimiter):
            yield line
    else:
        with open(filepath) as f:
            yield f.read().rstrip()


def create_translate_model(arg):

    if arg == "googletrans":
        # We do auto-import so we not depend on the actually installed library.
        translate_value = auto_import("arelight.third_party.googletrans.translate_value")
        # Translation of the list of data.
        # Returns the list of strings.
        return lambda str_list, src, dest: [translate_value(s, dest=dest, src=src) for s in str_list]


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


def input_to_docs(input_data, sentence_parser, docs_limit=None):
    """ input_data: list
        sentence_splitter: object
            how data is suppose to be separated onto sentences.
            str -> list(str)
    """
    assert(input_data is not None)
    assert(isinstance(docs_limit, int) or docs_limit is None)

    docs = []

    for doc_ind, contents in enumerate(input_data):

        # setup input data.
        sentences = sentence_parser(contents)
        sentences = list(map(lambda text: BaseDocumentSentence(text), sentences))

        # Documents.
        docs.append(Document(doc_id=doc_ind, sentences=sentences))

        # Optionally checking for the limit.
        if docs_limit is not None and doc_ind >= docs_limit:
            break

    return docs


def get_list_choice(op_list):
    while True:
        choice = input("Select:\n{ops}\n".format(
            ops="\n".join(["{}: {}".format(i, n) for i, n in enumerate(op_list)])
        ))
        try:
            choice = int(choice)
            if 0 <= choice < len(op_list):
                break
            else:
                print("Invalid choice.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return op_list[choice]


def get_binary_choice(prompt):
    while True:
        choice = input(prompt).lower()
        if isinstance(choice, str) and len(choice) == 1:
            return choice == 'y'
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def get_int_choice(prompt, filter_func, is_optional=False):
    while True:
        choice = input(prompt).lower()
        if (is_optional and choice is None) or (isinstance(choice, int) and filter_func(choice)):
            return choice
        else:
            print("Invalid input. Please enter number.")


def is_port_number(number, is_optional=True):
    if is_optional and not number:
        return True
    return 1 < int(number) < 65535


OPENNRE_CHECKPOINTS = {
    "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar": {
        "state": "DeepPavlov/rubert-base-cased",
        "checkpoint": "https://www.dropbox.com/scl/fi/rwjf7ag3w3z90pifeywrd/ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar?rlkey=p0mmu81o6c2u6iboe9m20uzqk&dl=1",
        "label_scaler": CustomLabelScaler(p=1, n=2, u=0)
    },
    "ra4-rsr1_bert-base-cased_cls.pth.tar": {
        "state": "bert-base-cased",
        "checkpoint": "https://www.dropbox.com/scl/fi/k5arragv1g4wwftgw5xxd/ra-rsr_bert-base-cased_cls.pth.tar?rlkey=8hzavrxunekf0woesxrr0zqys&dl=1",
        "label_scaler": CustomLabelScaler(p=1, n=2, u=0)
    }
}

import importlib

from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arelight.run.entities.types import EntityFormattersService
from arelight.run.utils import create_sentence_parser
from arelight.samplers.types import SampleFormattersService


class BaseArg:

    @staticmethod
    def read_argument(args):
        raise NotImplementedError()

    @staticmethod
    def add_argument(parser, default):
        raise NotImplementedError()


class InputTextArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.input_text

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--text',
                            dest='input_text',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Input text for processing')


class FromDataframeArg(BaseArg):

    @staticmethod
    def read_argument(args):

        if args.from_dataframe is None:
            return None

        path = args.from_dataframe[0]

        if path is None:
            return None

        # loading from CSV. file
        if ".csv" in path:
            pd = importlib.import_module("pandas")
            df = pd.read_csv(path, delimiter=",")
            return df["text"].astype(str).to_list()

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--from-dataframe',
                            dest='from_dataframe',
                            type=str,
                            default=default,
                            nargs=1)


class FromFilesArg(BaseArg):

    @staticmethod
    def read_argument(args):
        paths = args.from_files

        if paths is None:
            return None

        file_contents = []
        for path in paths:
            with open(path) as f:
                file_contents.append(f.read().rstrip())

        return file_contents

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--from-files',
                            dest='from_files',
                            type=str,
                            default=default,
                            nargs='+',
                            help='Custom vocabulary filepath')


class StemmerArg(BaseArg):

    supported = {
        u"mystem": MystemWrapper()
    }

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return StemmerArg.supported[args.stemmer]

    @staticmethod
    def add_argument(parser, default):
        assert(default in StemmerArg.supported)
        parser.add_argument('--stemmer',
                            dest='stemmer',
                            type=str,
                            choices=list(StemmerArg.supported.keys()),
                            default=default,
                            nargs='?',
                            help='Stemmer (Default: {})'.format(default))


class TermsPerContextArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.terms_per_context

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--terms-per-context',
                            dest='terms_per_context',
                            type=int,
                            default=default,
                            nargs='?',
                            help='The max possible length of an input context in terms (Default: {})\n'
                                 'NOTE: Use greater or equal value for this parameter during experiment'
                                 'process; otherwise you may encounter with exception during sample '
                                 'creation process!'.format(default))


class SynonymsCollectionFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.synonyms_filepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--synonyms-filepath',
                            dest='synonyms_filepath',
                            type=str,
                            default=default,
                            help="List of synonyms provided in lines of the source text file.")


class SentenceParserArg(BaseArg):

    @staticmethod
    def read_argument(args):
        arg = args.sentence_parser
        return create_sentence_parser(arg)

    @staticmethod
    def add_argument(parser, default="ru"):
        parser.add_argument('--sentence-parser',
                            dest='sentence_parser',
                            type=str,
                            choices=['linesplit', 'ru', 'nltk_en'],
                            default=default)


class NERModelNameArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.ner_model_name

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--ner-model-name',
                            dest='ner_model_name',
                            type=str,
                            default=default)


class NERObjectTypes(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.ner_types.split("|")

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--ner-types',
                            dest='ner_types',
                            type=str,
                            default=default,
                            help="Filters specific NER types; provide with `|` separator")

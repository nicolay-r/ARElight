import importlib

from arelight.run.utils import create_sentence_parser


class BaseArg:

    @staticmethod
    def read_argument(args):
        raise NotImplementedError()

    @staticmethod
    def add_argument(parser, default):
        raise NotImplementedError()


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

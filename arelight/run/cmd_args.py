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


class OutputFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.inference_output_filepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('-o',
                            dest='inference_output_filepath',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Inference output filepath')


class VocabFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.vocab_filepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--vocab-filepath',
                            dest='vocab_filepath',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Custom vocabulary filepath')


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


class UseBalancingArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.balance_samples

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, bool))
        parser.add_argument('--balance-samples',
                            dest='balance_samples',
                            type=lambda x: (str(x).lower() == 'true'),
                            default=str(default),
                            nargs=1,
                            help='Use balancing for Train type during sample serialization process"')


class DistanceInTermsBetweenAttitudeEndsArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.dist_between_ends

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--dist-between-att-ends',
                            dest='dist_between_ends',
                            type=int,
                            default=default,
                            nargs='?',
                            help='Distance in terms between attitude participants in terms.'
                                 '(Default: {})'.format(None))


class EmbeddingMatrixFilepathArg(BaseArg):
    """ Embedding matrix, utilized as an input for model.
    """

    @staticmethod
    def read_argument(args):
        return args.embedding_matrix_filepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--emb-npz-filepath',
                            dest='embedding_matrix_filepath',
                            type=str,
                            default=default,
                            help='RusVectores embedding filepath')


class LabelsCountArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.labels_count

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--labels-count',
                            dest="labels_count",
                            type=int,
                            default=default,
                            help="Labels count in an output classifier")


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


class TokensPerContextArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.tokens_per_context

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--tokens-per-context',
                            dest='tokens_per_context',
                            type=int,
                            default=default,
                            nargs='?')


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


class DoLowercaseArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.do_lowercase

    @staticmethod
    def add_argument(parser, default):
        assert (isinstance(default, int))
        parser.add_argument('--do-lowercase',
                            dest='do_lowercase',
                            type=bool,
                            default=default,
                            nargs='?')


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


class PretrainedBERTArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.pretrained_bert

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--pretrained-bert',
                            dest='pretrained_bert',
                            type=str,
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

class ModelLoadDirArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.model_load_dir

    @staticmethod
    def add_argument(parser, default=None):
        parser.add_argument('--model-state-dir',
                            dest='model_load_dir',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Use pretrained state as initial')


class EntityFormatterTypesArg(BaseArg):

    @staticmethod
    def read_argument(args):
        name = args.entity_fmt
        return EntityFormattersService.name_to_type(name)

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, str))
        assert(EntityFormattersService.is_supported(default))
        parser.add_argument('--entity-fmt',
                            dest='entity_fmt',
                            type=str,
                            choices=list(EntityFormattersService.iter_names()),
                            default=default,
                            help='Entity formatter type')


class BertSaveFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.bert_savepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--bert-savepath',
                            dest='bert_savepath',
                            type=str,
                            default=default,
                            help='Bert state save filepath')


class InputSamplesFilepath(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.input_samples

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--input-samples',
                            dest='input_samples',
                            type=str,
                            default=default,
                            help='Input Samples')


class InputSamplesDir(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.input_samples_dir

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--input-samples-dir',
                            dest='input_samples_dir',
                            type=str,
                            default=default,
                            help='Input Samples')


class BertTextBFormatTypeArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return SampleFormattersService.name_to_type(args.text_b_type)

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, str))
        assert(SampleFormattersService.is_supported(default))
        parser.add_argument('--text-b-type',
                            dest='text_b_type',
                            type=str,
                            default=default,
                            choices=list(SampleFormattersService.iter_names()),
                            help='TextB format type (Default: {})'.format(default))

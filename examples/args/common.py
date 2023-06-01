import importlib

from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersionsService, RuSentiFramesVersions
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arelight.pipelines.demo.labels.base import NegativeLabel, PositiveLabel
from arelight.pipelines.items.entities_default import TextEntitiesParser
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.samplers.types import SampleFormattersService

from examples.args.base import BaseArg
from examples.entities.types import EntityFormattersService


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


class FramesColectionArg(BaseArg):

    @staticmethod
    def read_argument(args):
        if args.frames == "ruattitudes-20":
            return RuSentiFramesCollection.read_collection(
                version=RuSentiFramesVersions.V20,
                labels_fmt=RuSentiFramesLabelsFormatter(
                    pos_label_type=PositiveLabel, neg_label_type=NegativeLabel),
                effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                    pos_label_type=PositiveLabel, neg_label_type=NegativeLabel))

    @staticmethod
    def add_argument(parser, default="ruattitudes-20"):
        parser.add_argument('--frames',
                            dest='frames',
                            type=str,
                            default=default,
                            nargs='?',
                            help='Collection for frames annotation in text (Default: {})'.format(default))


class PredictOutputFilepathArg(BaseArg):

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


class RuSentiFramesVersionArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return RuSentiFramesVersionsService.get_type_by_name(args.frames_version)

    @staticmethod
    def add_argument(parser, default=RuSentiFramesVersionsService.get_name_by_type(RuSentiFramesVersions.V20)):

        parser.add_argument('--frames-version',
                            dest='frames_version',
                            type=str,
                            default=default,
                            choices=list(RuSentiFramesVersionsService.iter_supported_names()),
                            nargs='?',
                            help='Version of RuSentiFrames collection (Default: {})'.format(default))


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


class SentenceParserArg(BaseArg):

    @staticmethod
    def read_argument(args):
        arg = args.sentence_parser
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

    @staticmethod
    def add_argument(parser, default="ru"):
        parser.add_argument('--sentence-parser',
                            dest='sentence_parser',
                            type=str,
                            choices=['linesplit', 'ru', 'nltk_en'],
                            default=default)


class EntitiesParserArg(BaseArg):

    @staticmethod
    def read_argument(args):
        arg = args.entities_parser
        if arg == "default":
            return TextEntitiesParser()
        elif arg == "bert-ontonotes":
            # We consider only such entity types that supported by ML model.
            ppl_item = DeepPavlovNERPipelineItem(
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_cfg="ontonotes_mult")
            return ppl_item
        elif arg == "bert-ontonotes-eng":
            # We consider only such entity types that supported by ML model.
            ppl_item = DeepPavlovNERPipelineItem(
                obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                ner_model_cfg="ontonotes_eng")
            return ppl_item

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--entities-parser',
                            dest='entities_parser',
                            type=str,
                            choices=['no', 'bert-ontonotes', 'bert-ontonotes-eng'],
                            default=default,
                            help='Adopt entities parser in text processing (default: {})'.format(default))


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


class BertConfigFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.bert_config

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--bert-config',
                            dest='bert_config',
                            type=str,
                            default=default,
                            help='Bert config filepath')


class BertCheckpointFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.bert_checkpoint

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--bert-checkpoint',
                            dest='bert_checkpoint',
                            type=str,
                            default=default,
                            help='Bert checkpoint filepath')


class BertVocabFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.bert_vocab

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--bert-vocab',
                            dest='bert_vocab',
                            type=str,
                            default=default,
                            help='Bert vocab filepath')


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

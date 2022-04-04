from arekit.contrib.experiment_rusentrel.entities.types import EntityFormattersService
from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import ExperimentRuSentiFramesLabelsFormatter
from arekit.contrib.experiment_rusentrel.synonyms.collection import StemmerBasedSynonymCollection
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.enum_name_types import ModelNamesService
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersionsService, RuSentiFramesVersions
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups
from arekit.processing.lemmatization.mystem import MystemWrapper

from network.args.base import BaseArg
from text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem
from text.pipeline_entities_default import TextEntitiesParser


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
                labels_fmt=ExperimentRuSentiFramesLabelsFormatter())

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
    def add_argument(parser, default):
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


class RusVectoresEmbeddingFilepathArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return args.embedding_filepath

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--emb-filepath',
                            dest='embedding_filepath',
                            type=str,
                            default=default,
                            help='RusVectores embedding filepath')


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


class SynonymsCollectionArg(BaseArg):

    @staticmethod
    def __iter_groups(filepath):
        with open(filepath, 'r') as file:
            for group in iter_synonym_groups(file):
                yield group

    @staticmethod
    def read_argument(args):
        filepath = args.synonyms_filepath
        stemmer = MystemWrapper()
        if filepath is None:
            # Provide RuSentRel collection by default.
            return RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer,
                                                                       version=RuSentRelVersions.V11,
                                                                       is_read_only=False)
        else:
            # Provide collection from file.
            return StemmerBasedSynonymCollection(iter_group_values_lists=SynonymsCollectionArg.__iter_groups(filepath),
                                                 stemmer=stemmer,
                                                 is_read_only=False,
                                                 debug=False)

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--synonyms-filepath',
                            dest='synonyms_filepath',
                            type=str,
                            default=None,
                            help="List of synonyms provided in lines of the source text file.")


class EntitiesParserArg(BaseArg):

    @staticmethod
    def read_argument(args):
        arg = args.entities_parser
        if arg == "default":
            return TextEntitiesParser()
        elif arg == "bert-ontonotes":
            # We consider only such entity types that supported by ML model.
            ppl_item = BertOntonotesNERPipelineItem(
                lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"])
            return ppl_item

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--entities-parser',
                            dest='entities_parser',
                            type=str,
                            choices=['no', 'bert-ontonotes'],
                            default=default,
                            help='Adopt entities parser in text processing (default: {})'.format(default))


class ModelNameArg(BaseArg):

    @staticmethod
    def read_argument(args):
        return ModelNamesService.get_type_by_name(args.model_name)

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--model-name',
                            dest='model_name',
                            type=str,
                            choices=list(ModelNamesService.iter_supported_names()),
                            default=default,
                            help='Name of a model to be utilized in experiment')


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

from arekit.contrib.networks.enum_input_types import ModelInputType, ModelInputTypeService

from network.args.base import BaseArg


class BagsPerMinibatchArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.bags_per_minibatch

    @staticmethod
    def add_argument(parser, default):
        parser.add_argument('--bags-per-minibatch',
                            dest='bags_per_minibatch',
                            type=int,
                            default=default,
                            nargs='?',
                            help='Bags per minibatch count (Default: {})'.format(default))


class DropoutKeepProbArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.dropout_keep_prob

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, float))
        parser.add_argument('--dropout-keep-prob',
                            dest='dropout_keep_prob',
                            type=float,
                            default=default,
                            nargs='?',
                            help='Dropout keep prob (Default: {})'.format(default))


class EpochsCountArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.epochs

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, int))
        parser.add_argument('--epochs',
                            dest='epochs',
                            type=int,
                            default=default,
                            nargs='?',
                            help='Epochs count (Default: {})'.format(default))


class BatchSizeArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.batch_size

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, int))
        parser.add_argument('--batch-size',
                            dest='batch_size',
                            type=int,
                            default=default,
                            nargs='?',
                            help='Batch size (Default: {})'.format(default))


class LearningRateArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.learning_rate

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, float))
        parser.add_argument('--learning-rate',
                            dest='learning_rate',
                            type=float,
                            default=default,
                            nargs='?',
                            help='Learning Rate (Default: {})'.format(default))


class ModelInputTypeArg(BaseArg):

    _default = ModelInputType.SingleInstance

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return ModelInputTypeService.get_type_by_name(args.input_type)

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, ModelInputType))
        parser.add_argument('--model-input-type',
                            dest='input_type',
                            type=str,
                            choices=list(ModelInputTypeService.iter_supported_names()),
                            default=ModelInputTypeService.find_name_by_type(default),
                            nargs='?',
                            help='Input format type (Default: {})'.format(default))


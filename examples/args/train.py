from examples.args.base import BaseArg


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


class DoLowercaseArg(BaseArg):

    def __init__(self):
        pass

    @staticmethod
    def read_argument(args):
        return args.do_lowercase

    @staticmethod
    def add_argument(parser, default):
        assert(isinstance(default, int))
        parser.add_argument('--do-lowercase',
                            dest='do_lowercase',
                            type=bool,
                            default=default,
                            nargs='?')


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

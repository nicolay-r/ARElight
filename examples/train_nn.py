import argparse

from examples.args import const, train
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames

from examples.args import common

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Training script for obtaining Tensorflow based states, "
                                                 "based on the RuSentRel and RuAttitudes datasets (optionally)")

    # Composing cmd arguments.
    common.LabelsCountArg.add_argument(parser, default=3)
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser, default=None)
    common.ModelNameArg.add_argument(parser, default=ModelNames.PCNN.value)
    common.VocabFilepathArg.add_argument(parser, default=None)
    common.EmbeddingMatrixFilepathArg.add_argument(parser, default=None)
    common.ModelLoadDirArg.add_argument(parser, default=None)
    train.ModelInputTypeArg.add_argument(parser, default=ModelInputType.SingleInstance)
    train.BagsPerMinibatchArg.add_argument(parser, default=const.BAGS_PER_MINIBATCH)
    train.DropoutKeepProbArg.add_argument(parser, default=0.5)
    train.LearningRateArg.add_argument(parser, default=0.1)
    train.EpochsCountArg.add_argument(parser, default=150)

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    labels_count = common.LabelsCountArg.read_argument(args)
    model_name = common.ModelNameArg.read_argument(args)
    embedding_matrix_filepath = common.EmbeddingMatrixFilepathArg.read_argument(args)
    vocab_filepath = common.VocabFilepathArg.read_argument(args)
    terms_per_context = common.TermsPerContextArg.read_argument(args)
    dist_in_terms_between_attitude_ends = common.DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)
    model_load_dir = common.ModelLoadDirArg.read_argument(args)
    model_input_type = train.ModelInputTypeArg.read_argument(args)
    bags_per_minibatch = train.BagsPerMinibatchArg.read_argument(args)
    dropout_keep_prob = train.DropoutKeepProbArg.read_argument(args)
    learning_rate = train.LearningRateArg.read_argument(args)
    epochs_count = train.EpochsCountArg.read_argument(args)

    # TODO. Adopt code from the NIVTS Project.
    # TODO.
    # TODO.
    # TODO.
    # TODO.

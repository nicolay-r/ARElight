import argparse

from arekit.contrib.utils.np_utils.writer import NpzDataWriter
from arekit.contrib.utils.processing.languages.ru.pos_service import PartOfSpeechTypesService

from arelight.network.nn.common import create_bags_collection_type, create_full_model_name, create_network_model_io

from examples.args import const, train
from examples.rusentrel.common import Common
from examples.args.const import NEURAL_NETWORKS_TARGET_DIR, BAG_SIZE
from examples.rusentrel.config_setups import optionally_modify_config_for_experiment, modify_config_for_model
from examples.rusentrel.exp_io import CustomRuSentRelNetworkExperimentIO

from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.types import FoldingType
from arekit.contrib.networks.context.configurations.base.base import DefaultNetworkConfig
from arekit.contrib.networks.core.callback.hidden import HiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

from examples.args import common
from examples.utils import create_labels_scaler

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

    # Utilize predefined versions and folding format.
    # TODO. This is outdated.
    exp_type = ExperimentTypes.RuSentRel
    rusentrel_version = RuSentRelVersions.V11
    folding_type = FoldingType.Fixed
    model_target_dir = NEURAL_NETWORKS_TARGET_DIR

    # Init handler.
    bags_collection_type = create_bags_collection_type(model_input_type=model_input_type)
    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=model_input_type)

    labels_scaler = create_labels_scaler(labels_count)

    exp_name = Common.create_exp_name(rusentrel_version=rusentrel_version,
                                      ra_version=None,
                                      folding_type=folding_type)

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=True,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    data_folding = Common.create_folding(
        rusentrel_version=rusentrel_version,
        ruattitudes_version=None,
        doc_id_func=lambda doc_id: Common.ra_doc_id_func(doc_id=doc_id))

    # Creating experiment
    exp_ctx = ExperimentTrainingContext(labels_count=labels_scaler.LabelsCount,
                                        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix),
                                        data_folding=data_folding)

    exp_io = CustomRuSentRelNetworkExperimentIO(exp_ctx)

    full_model_name = create_full_model_name(model_name=model_name,
                                             input_type=model_input_type)

    model_io = create_network_model_io(full_model_name=full_model_name,
                                       source_dir=model_load_dir,
                                       target_dir=model_target_dir,
                                       embedding_filepath=embedding_matrix_filepath,
                                       vocab_filepath=vocab_filepath,
                                       model_name_tag=u'')

    # Setup model io.
    exp_ctx.set_model_io(model_io)

    ###################
    # Initialize config
    ###################
    config = network_config_func()

    assert(isinstance(config, DefaultNetworkConfig))

    # Default settings, applied from cmd arguments.
    config.modify_classes_count(value=labels_count)
    config.modify_learning_rate(learning_rate)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(dropout_keep_prob)
    config.modify_bag_size(BAG_SIZE)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.modify_embedding_dropout_keep_prob(1.0)
    config.modify_terms_per_context(terms_per_context)
    config.modify_use_entity_types_in_embedding(False)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())

    # Modify config parameters. This may affect
    # the settings, already applied above!
    optionally_modify_config_for_experiment(exp_type=exp_type,
                                            model_input_type=model_input_type,
                                            config=config)

    # Modify config parameters. This may affect
    # the settings, already applied above!
    modify_config_for_model(model_name=model_name,
                            model_input_type=model_input_type,
                            config=config)

    data_writer = NpzDataWriter()

    nework_callbacks = [
        TrainingLimiterCallback(train_acc_limit=0.99),
        TrainingStatProviderCallback(),
        HiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer),
        InputHiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer)
    ]

    # TODO. Switch to pipeline.
    training_handler = NetworksTrainingIterationHandler(
        load_model=model_load_dir is not None,
        exp_ctx=exp_ctx,
        exp_io=exp_io,
        create_network_func=network_func,
        config=config,
        bags_collection_type=bags_collection_type,
        network_callbacks=nework_callbacks,
        training_epochs=epochs_count)

    # TODO. Engine does not exists anymore.
    engine = ExperimentEngine(exp_ctx.DataFolding)

    engine.run(handlers=[training_handler])

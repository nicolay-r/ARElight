import argparse
from os.path import join

from arekit.common.data.input.readers.tsv import TsvReader
from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.callback.hidden import HiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.hidden_input import InputHiddenStatesWriterCallback
from arekit.contrib.networks.core.callback.stat import TrainingStatProviderCallback
from arekit.contrib.networks.core.callback.train_limiter import TrainingLimiterCallback
from arekit.contrib.networks.core.feeding.bags.collection.single import SingleBagsCollection
from arekit.contrib.networks.core.model_io import NeuralNetworkModelIO
from arekit.contrib.networks.factory import create_network_and_network_config_funcs
from arekit.contrib.networks.pipelines.items.training import NetworksTrainingPipelineItem
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.np_utils.writer import NpzDataWriter
from arekit.contrib.utils.processing.languages.ru.pos_service import PartOfSpeechTypesService

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
    common.InputSamplesDir.add_argument(parser, default=join(const.OUTPUT_DIR, "rsr-v1_1-fx-balanced-tpc50_3l"))
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
    input_samples_dir = common.InputSamplesDir.read_argument(args)

    folding_type = "fixed"
    output_dir = "out"
    model_log_dir = ""
    train_acc_limit = 0.99
    finetune_existed = True
    full_model_name = "-".join([folding_type, model_name.value])
    model_target_dir = join(model_log_dir, full_model_name)

    model_io = NeuralNetworkModelIO(full_model_name=full_model_name,
                                    target_dir=output_dir,
                                    source_dir=output_dir if finetune_existed else None,
                                    model_name_tag=u'')

    data_writer = NpzDataWriter()

    network_func, network_config_func = create_network_and_network_config_funcs(
        model_name=model_name,
        model_input_type=ModelInputType.SingleInstance)

    network_callbacks = [
        TrainingLimiterCallback(train_acc_limit=train_acc_limit),
        TrainingStatProviderCallback(),
        HiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer),
        InputHiddenStatesWriterCallback(log_dir=model_target_dir, writer=data_writer)
    ]

    bag_size = 1
    embedding_dropout_keep_prob = 1.0

    # Configuration initialization.
    config = network_config_func()
    config.modify_classes_count(value=labels_count)
    config.modify_learning_rate(learning_rate)
    config.modify_use_class_weights(True)
    config.modify_dropout_keep_prob(dropout_keep_prob)
    config.modify_bag_size(bag_size)
    config.modify_bags_per_minibatch(bags_per_minibatch)
    config.modify_embedding_dropout_keep_prob(embedding_dropout_keep_prob)
    config.modify_terms_per_context(terms_per_context)
    config.modify_use_entity_types_in_embedding(False)
    config.set_pos_count(PartOfSpeechTypesService.get_mystem_pos_count())

    pipeline_item = NetworksTrainingPipelineItem(
        load_model=True,
        model_io=model_io,
        labels_count=labels_count,
        create_network_func=network_func,
        samples_io=SamplesIO(target_dir=input_samples_dir, reader=TsvReader()),
        emb_io=NpEmbeddingIO(target_dir=input_samples_dir),
        config=config,
        bags_collection_type=SingleBagsCollection,
        network_callbacks=network_callbacks,
        training_epochs=epochs_count)

    # Start training process.
    ppl = BasePipeline([pipeline_item])
    ppl.run(None, params_dict={"data_folding": NoFolding(doc_ids=[], supported_data_type=DataType.Train),
                               "data_type": DataType.Train})

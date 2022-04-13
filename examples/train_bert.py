import argparse
from os.path import join

from arekit.common.pipeline.base import BasePipeline
from arelight.pipelines.train_bert import BertFinetunePipelineItem

from examples.args import train, common, const

if __name__ == '__main__':

    # Setup parser.
    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    common.TokensPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.BertConfigFilepathArg.add_argument(parser, default=const.BERT_CONFIG_PATH)
    common.BertCheckpointFilepathArg.add_argument(parser, default=const.BERT_CKPT_PATH)
    common.BertVocabFilepathArg.add_argument(parser, default=const.BERT_VOCAB_PATH)
    common.BertSaveFilepathArg.add_argument(parser, default=join(const.BERT_TARGET_DIR, const.BERT_DEFAULT_STATE_NAME))
    common.InputSamplesFilepath.add_argument(parser, default=join(const.OUTPUT_DIR, join("rsr-v1_1-fx-nobalance-tpc50-bert_3l", "sample-train-0.tsv.gz")))
    train.LearningRateArg.add_argument(parser, default=2e-5)
    train.EpochsCountArg.add_argument(parser, default=4)
    train.BatchSizeArg.add_argument(parser, default=6)
    train.DoLowercaseArg.add_argument(parser, default=False)

    # Parsing arguments.
    args = parser.parse_args()

    # Compose pipeline item.
    ppl = BasePipeline([
        BertFinetunePipelineItem(bert_config_file=common.BertConfigFilepathArg.read_argument(args),
                                 model_checkpoint_path=common.BertCheckpointFilepathArg.read_argument(args),
                                 vocab_filepath=common.BertVocabFilepathArg.read_argument(args),
                                 do_lowercase=train.DoLowercaseArg.read_argument(args),
                                 max_seq_length=common.TokensPerContextArg.read_argument(args),
                                 learning_rate=train.LearningRateArg.read_argument(args),
                                 save_path=common.BertSaveFilepathArg.read_argument(args))
    ])

    ppl.run(common.InputSamplesFilepath.read_argument(args),
            params_dict={"epochs_count": train.EpochsCountArg.read_argument(args),
                         "batch_size": train.BatchSizeArg.read_argument(args)})

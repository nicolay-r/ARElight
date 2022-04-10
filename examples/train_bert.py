import argparse
import sys
from os.path import join

sys.path.append('../')

from arekit.common.pipeline.base import BasePipeline
from network.args import const
from network.args.common import BertConfigFilepathArg, BertCheckpointFilepathArg, BertVocabFilepathArg, \
    BertSaveFilepathArg, InputSamplesFilepath, TokensPerContextArg
from network.args.const import BERT_CONFIG_PATH, BERT_CKPT_PATH, BERT_VOCAB_PATH, OUTPUT_DIR, \
    BERT_DEFAULT_STATE_NAME, BERT_TARGET_DIR
from network.args.train import EpochsCountArg, BatchSizeArg, LearningRateArg, DoLowercaseArg
from pipelines.train_bert import BertFinetunePipelineItem

if __name__ == '__main__':

    # Setup parser.
    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    TokensPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    BertConfigFilepathArg.add_argument(parser, default=BERT_CONFIG_PATH)
    BertCheckpointFilepathArg.add_argument(parser, default=BERT_CKPT_PATH)
    BertVocabFilepathArg.add_argument(parser, default=BERT_VOCAB_PATH)
    BertSaveFilepathArg.add_argument(parser, default=join(BERT_TARGET_DIR, BERT_DEFAULT_STATE_NAME))
    InputSamplesFilepath.add_argument(parser, default=join(OUTPUT_DIR, join("rsr-v1_1-fx-nobalance-tpc50-bert_3l", "sample-train-0.tsv.gz")))
    LearningRateArg.add_argument(parser, default=2e-5)
    EpochsCountArg.add_argument(parser, default=4)
    BatchSizeArg.add_argument(parser, default=6)
    DoLowercaseArg.add_argument(parser, default=False)

    # Parsing arguments.
    args = parser.parse_args()

    # Compose pipeline item.
    ppl = BasePipeline([
        BertFinetunePipelineItem(bert_config_file=BertConfigFilepathArg.read_argument(args),
                                 model_checkpoint_path=BertCheckpointFilepathArg.read_argument(args),
                                 vocab_filepath=BertVocabFilepathArg.read_argument(args),
                                 do_lowercase=DoLowercaseArg.read_argument(args),
                                 max_seq_length=TokensPerContextArg.read_argument(args),
                                 learning_rate=LearningRateArg.read_argument(args),
                                 save_path=BertSaveFilepathArg.read_argument(args))
    ])

    ppl.run(InputSamplesFilepath.read_argument(args),
            params_dict={"epochs_count": EpochsCountArg.read_argument(args),
                         "batch_size": BatchSizeArg.read_argument(args)})

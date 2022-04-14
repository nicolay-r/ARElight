from os.path import join, dirname, realpath

current_dir = dirname(realpath(__file__))

# Predefined default parameters.
TERMS_PER_CONTEXT = 50
BAGS_PER_MINIBATCH = 2
BAG_SIZE = 1

DATA_DIR = join(current_dir, "../../data")
DEFAULT_TEXT_FILEPATH = join(DATA_DIR, "texts-inosmi-rus/e1.txt")
EMBEDDING_FILEPATH = join(DATA_DIR, "news_mystem_skipgram_1000_20_2015.bin.gz")
SYNONYMS_FILEPATH = join(DATA_DIR, "synonyms.txt")

# Common model dir.
DEFAULT_MODEL_DIR = join(DATA_DIR, "models")

PCNN_DEFAULT_MODEL_TAR = join(DEFAULT_MODEL_DIR, "fx_ctx_pcnn.tar.gz")

# Default pretrained BERT.
NEURAL_NETWORKS_TARGET_DIR = DEFAULT_MODEL_DIR
BERT_DEFAULT_STATE_NAME = "ra-20-srubert-large-neut-nli-pretrained-3l"
BERT_PRETRAINED_MODEL_PATHDIR = join(NEURAL_NETWORKS_TARGET_DIR, BERT_DEFAULT_STATE_NAME)
BERT_PRETRAINED_MODEL_TAR = BERT_PRETRAINED_MODEL_PATHDIR + '.tar.gz'
BERT_CONFIG_PATH = join(BERT_PRETRAINED_MODEL_PATHDIR, "bert_config.json")
BERT_CKPT_PATH = join(BERT_PRETRAINED_MODEL_PATHDIR, "model.ckpt-30238")
BERT_VOCAB_PATH = join(BERT_PRETRAINED_MODEL_PATHDIR, "vocab.txt")

# Default Fine-tuned BERT.
BERT_TARGET_DIR = DEFAULT_MODEL_DIR
BERT_DEFAULT_FINETUNED = BERT_DEFAULT_STATE_NAME + '-finetuned'
BERT_FINETUNED_MODEL_PATHDIR = join(NEURAL_NETWORKS_TARGET_DIR, BERT_DEFAULT_FINETUNED)
BERT_FINETUNED_MODEL_TAR = BERT_FINETUNED_MODEL_PATHDIR + '.tar.gz'
BERT_FINETUNED_CKPT_PATH = join(BERT_FINETUNED_MODEL_PATHDIR, BERT_DEFAULT_STATE_NAME)

# The common output directory.
OUTPUT_DIR = join(current_dir, "../../_output")

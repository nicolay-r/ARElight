from os.path import join, dirname, realpath

current_dir = dirname(realpath(__file__))

# Predefined default parameters.
TERMS_PER_CONTEXT = 50
BAGS_PER_MINIBATCH = 2
BAG_SIZE = 1

DATA_DIR = join(current_dir, "../../data")
DEFAULT_TEXT_FILEPATH = join(DATA_DIR, "texts-inosmi-rus/e2.txt")
EMBEDDING_FILEPATH = join(DATA_DIR, "news_mystem_skipgram_1000_20_2015.bin.gz")
SYNONYMS_FILEPATH = join(DATA_DIR, "synonyms.txt")

NEURAL_NETWORKS_TARGET_DIR = join(current_dir, "../../models/")
BERT_DEFAULT_STATE = "ra-20-srubert-large-neut-nli-pretrained-3l"
BERT_MODEL_PATH = join(NEURAL_NETWORKS_TARGET_DIR, BERT_DEFAULT_STATE)
BERT_CONFIG_PATH = join(BERT_MODEL_PATH, "bert_config.json")
BERT_CKPT_PATH = join(BERT_MODEL_PATH, "model.ckpt-30238")
BERT_VOCAB_PATH = join(BERT_MODEL_PATH, "vocab.txt")

OUTPUT_DIR = join(current_dir, "../../_output")

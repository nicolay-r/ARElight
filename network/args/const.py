from os.path import join, dirname, realpath

current_dir = dirname(realpath(__file__))

# Predefined default parameters.
TERMS_PER_CONTEXT = 50
BAGS_PER_MINIBATCH = 2
BAG_SIZE = 1
DATA_DIR = join(current_dir, "../../data")
EMBEDDING_FILEPATH = join(current_dir, "../../models", "news_mystem_skipgram_1000_20_2015.bin.gz")
NEURAL_NETWORKS_TARGET_DIR = DATA_DIR

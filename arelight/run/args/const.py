from os.path import join, dirname, realpath

current_dir = dirname(realpath(__file__))

# Predefined default parameters.
TERMS_PER_CONTEXT = 50
BAGS_PER_MINIBATCH = 2
BAG_SIZE = 1
DEFAULT_ENTITIES_PARSER = "bert-ontonotes"

DATA_DIR = join(current_dir, "../../../data")
SYNONYMS_FILEPATH = join(DATA_DIR, "synonyms.txt")

# The common output directory.
OUTPUT_TEMPLATE = join(current_dir, "../../_output", "sample")

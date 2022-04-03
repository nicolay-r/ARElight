from os.path import join
from arekit.contrib.source import utils
from network.args.const import EMBEDDING_FILEPATH, SYNONYMS_FILEPATH, BERT_MODEL_PATH


def download_examples_data():
    root_dir = utils.get_default_download_dir()

    data = {
        EMBEDDING_FILEPATH: "http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz",
        SYNONYMS_FILEPATH: "https://raw.githubusercontent.com/nicolay-r/RuSentRel/v1.1/synonyms.txt",
        BERT_MODEL_PATH: "https://www.dropbox.com/s/cr6nejxjiqbyd5o/ra-20-srubert-large-neut-nli-pretrained-3l.tar.gz?dl=1"
    }

    # Perform downloading ...
    for local_name, url_link in data.items():
        print("Downloading: {}".format(local_name))
        utils.download(dest_file_path=join(root_dir, local_name),
                       source_url=url_link)


if __name__ == '__main__':
    download_examples_data()

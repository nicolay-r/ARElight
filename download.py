import os
import tarfile
from arekit.contrib.source import utils

from examples.args import const


def download_examples_data():
    root_dir = utils.get_default_download_dir()

    data = {
        const.EMBEDDING_FILEPATH: "http://rusvectores.org/static/models/rusvectores2/news_mystem_skipgram_1000_20_2015.bin.gz",
        const.SYNONYMS_FILEPATH: "https://raw.githubusercontent.com/nicolay-r/RuSentRel/v1.1/synonyms.txt",
        # PCNN: pretrained model dir.
        const.PCNN_DEFAULT_MODEL_TAR: "https://www.dropbox.com/s/ceqy69vj59te534/fx_ctx_pcnn.tar.gz?dl=1",
        # NOTE: this is a pre-trained model and it is expected to be fine-tunned.
        const.BERT_PRETRAINED_MODEL_TAR: "https://www.dropbox.com/s/cr6nejxjiqbyd5o/ra-20-srubert-large-neut-nli-pretrained-3l.tar.gz?dl=1",
        # Fine-tuned on RuSentRel collection.
        const.BERT_FINETUNED_MODEL_TAR: "https://www.dropbox.com/s/g73osmwyrqtr2at/ra-20-srubert-large-neut-nli-pretrained-3l-finetuned.tar.gz?dl=1"
    }

    # Perform downloading ...
    for local_name, url_link in data.items():
        print("Downloading: {}".format(local_name))
        utils.download(dest_file_path=os.path.join(root_dir, local_name),
                       source_url=url_link)

    # Extracting tar files ...
    for local_name in data.keys():
        print(local_name)
        if not os.path.exists(local_name):
            continue
        if not tarfile.is_tarfile(local_name):
            continue
        with tarfile.open(local_name) as f:
            target = os.path.dirname(local_name)
            f.extractall(path=target)


if __name__ == '__main__':
    download_examples_data()

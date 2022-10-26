import os
import tarfile

from arekit.common import utils

from examples.args import const


def download_examples_data():
    root_dir = utils.get_default_download_dir()

    data = {
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
        dest_filepath = os.path.join(root_dir, local_name)
        if not os.path.exists(dest_filepath):
            continue
        if not tarfile.is_tarfile(dest_filepath):
            continue
        with tarfile.open(dest_filepath) as f:
            target = os.path.dirname(dest_filepath)
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(f, path=target)

        # Remove .tar file
        os.remove(dest_filepath)


if __name__ == '__main__':
    download_examples_data()

import os
import tarfile
from os.path import dirname, realpath, join

from arekit.common import utils

current_dir = dirname(realpath(__file__))
DATA_DIR = join(current_dir, "./data")


def download_examples_data():
    root_dir = utils.get_default_download_dir()

    data = {
        join(DATA_DIR, "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar"):
            "https://www.dropbox.com/scl/fi/rwjf7ag3w3z90pifeywrd/ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar?rlkey=p0mmu81o6c2u6iboe9m20uzqk&dl=1",
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

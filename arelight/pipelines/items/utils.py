from os.path import join, exists

from arekit.common.utils import download

from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.run.utils import logger


def try_download_predefined_checkpoint(checkpoint, dir_to_download):
    """ This is for the simplicity of using the framework straightaway.
    """
    assert(isinstance(checkpoint, str))
    assert(isinstance(dir_to_download, str))

    predefined = {
        "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar": {
            "state": "DeepPavlov/rubert-base-cased",
            "checkpoint": "https://www.dropbox.com/scl/fi/rwjf7ag3w3z90pifeywrd/ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar?rlkey=p0mmu81o6c2u6iboe9m20uzqk&dl=1",
            "label_scaler": CustomLabelScaler(p=1, n=2, u=0)
        },
        "ra4-rsr1_bert-base-cased_cls.pth.tar": {
            "state": "bert-base-cased",
            "checkpoint": "https://www.dropbox.com/scl/fi/k5arragv1g4wwftgw5xxd/ra-rsr_bert-base-cased_cls.pth.tar?rlkey=8hzavrxunekf0woesxrr0zqys&dl=1",
            "label_scaler": CustomLabelScaler(p=1, n=2, u=0)
        }
    }

    if checkpoint in predefined:
        data = predefined[checkpoint]
        target_checkpoint_path = join(dir_to_download, checkpoint)

        logger.info("Found predefined checkpoint: {}".format(checkpoint))
        # No need to do anything, file has been already downloaded.
        if not exists(target_checkpoint_path):
            logger.info("Downloading checkpoint to: {}".format(target_checkpoint_path))
            download(dest_file_path=target_checkpoint_path, source_url=data["checkpoint"])

        return data["state"], target_checkpoint_path, data["label_scaler"]

    return None, None, None

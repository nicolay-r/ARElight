from os.path import join, exists

from arekit.common.utils import download
from arekit.common.docs.base import Document
from arekit.common.docs.sentence import BaseDocumentSentence

from arelight.run.utils import logger


def input_to_docs(input_data, sentence_parser, docs_limit=None):
    """ input_data: list
        sentence_splitter: object
            how data is suppose to be separated onto sentences.
            str -> list(str)
    """
    assert(input_data is not None)
    assert(isinstance(docs_limit, int) or docs_limit is None)

    docs = []

    for doc_ind, contents in enumerate(input_data):

        # setup input data.
        sentences = sentence_parser(contents)
        sentences = list(map(lambda text: BaseDocumentSentence(text), sentences))

        # Documents.
        docs.append(Document(doc_id=doc_ind, sentences=sentences))

        # Optionally checking for the limit.
        if docs_limit is not None and doc_ind >= docs_limit:
            break

    return docs


def try_download_predefined_checkpoint(checkpoint, dir_to_download):
    """ This is for the simplicity of using the framework straightaway.
    """
    assert(isinstance(checkpoint, str))
    assert(isinstance(dir_to_download, str))

    predefined = {
        "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar": {
            "state": "DeepPavlov/rubert-base-cased",
            "checkpoint": "https://www.dropbox.com/scl/fi/rwjf7ag3w3z90pifeywrd/ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar?rlkey=p0mmu81o6c2u6iboe9m20uzqk&dl=1",
        },
        "ra4-rsr1_bert-base-cased_cls.pth.tar": {
            "state": "bert-base-cased",
            "checkpoint": "https://www.dropbox.com/scl/fi/k5arragv1g4wwftgw5xxd/ra-rsr_bert-base-cased_cls.pth.tar?rlkey=8hzavrxunekf0woesxrr0zqys&dl=1"
        }
    }

    if checkpoint in predefined:
        data = predefined[checkpoint]
        target_path = join(dir_to_download, checkpoint)

        logger.info("Found predefined checkpoint: {}".format(checkpoint))
        # No need to do anything, file has been already downloaded.
        if not exists(target_path):
            logger.info("Downloading checkpoint to: {}".format(target_path))
            download(dest_file_path=target_path, source_url=data["checkpoint"])

        return data["state"], data["checkpoint"]

    return None, None

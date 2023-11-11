from os.path import dirname, realpath, join

from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.docs.base import Document
from arekit.common.docs.sentence import BaseDocumentSentence
from ru_sent_tokenize import ru_sent_tokenize

current_dir = dirname(realpath(__file__))
TEST_DATA_DIR = join(current_dir, "data")
TEST_OUT_DIR = join("_out")


class InMemoryDocProvider(DocumentProvider):

    def __init__(self, docs=None):
        assert(isinstance(docs, list) or docs is None)
        self.__docs = docs

    def by_id(self, doc_id):
        return self.__docs[doc_id]

    def __len__(self):
        return len(self.__docs)


def input_to_docs(texts):
    assert(isinstance(texts, list))
    docs = []
    for doc_id, contents in enumerate(texts):
        sentences = ru_sent_tokenize(contents)
        sentences = list(map(lambda text: BaseDocumentSentence(text), sentences))
        doc = Document(doc_id=doc_id, sentences=sentences)
        docs.append(doc)
    return docs

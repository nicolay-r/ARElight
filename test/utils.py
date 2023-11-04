from os.path import dirname, realpath, join

from arekit.common.data.doc_provider import DocumentProvider

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

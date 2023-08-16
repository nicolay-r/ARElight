from arekit.common.data.doc_provider import DocumentProvider


class InMemoryDocOperations(DocumentProvider):

    def __init__(self, docs=None):
        assert(isinstance(docs, list) or docs is None)
        self.__docs = docs

    def by_id(self, doc_id):
        return self.__docs[doc_id]

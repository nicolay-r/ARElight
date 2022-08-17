from arekit.common.experiment.api.ops_doc import DocumentOperations


class InMemoryDocOperations(DocumentOperations):

    def __init__(self, docs=None):
        assert(isinstance(docs, list) or docs is None)
        self.__docs = docs

    def set_docs(self, docs):
        assert(isinstance(docs, list))
        self.__docs = docs

    def get_doc(self, doc_id):
        return self.__docs[doc_id]

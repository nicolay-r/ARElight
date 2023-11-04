from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.docs.base import Document
from arekit.common.docs.sentence import BaseDocumentSentence


class CachedFilesDocProvider(DocumentProvider):

    def __init__(self, filepaths, content_provider, content_to_sentences):
        assert(callable(content_provider))
        assert(callable(content_to_sentences))

        self.__filepaths = filepaths

        self.__cached_content = None
        self.__cached_filename = None

        self.__content_provider = content_provider
        self.__cont_to_sent = content_to_sentences

    def get_file_content(self, filepath):

        if self.__cached_filename != filepath:
            self.__cached_content = list(self.__content_provider(filepath))
            self.__cached_filename = filepath

        return self.__cached_content

    def by_id(self, doc_id):
        assert(isinstance(doc_id, str))

        filename, row_index = doc_id.split(':') if ":" in doc_id else (doc_id, 0)
        content = self.get_file_content(filepath=filename)[int(row_index)]
        sentences = self.__cont_to_sent(content)

        return Document(doc_id=doc_id, sentences=list(map(lambda text: BaseDocumentSentence(text), sentences)))

    def iter_doc_ids(self):
        for filepath in self.__filepaths:
            for doc_ind, _ in enumerate(self.get_file_content(filepath)):
                yield ":".join([filepath, str(doc_ind)])

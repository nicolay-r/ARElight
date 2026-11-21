from arekit.common.data.doc_provider import DocumentProvider
from arekit.common.docs.base import Document
from arekit.common.docs.sentence import BaseDocumentSentence


class CachedFilesDocProvider(DocumentProvider):

    def __init__(self, filepaths, content_provider, content_to_sentences, docs_limit=None):
        assert(callable(content_provider))
        assert(callable(content_to_sentences))
        assert(isinstance(docs_limit, int) or docs_limit is None)

        self.__filepaths = filepaths

        self.__cached_content = None
        self.__cached_filename = None

        self.__content_provider = content_provider
        self.__cont_to_sent = content_to_sentences
        self.__docs_limit = docs_limit

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

                if self.__docs_limit is not None and doc_ind >= self.__docs_limit:
                    break

                yield ":".join([filepath, str(doc_ind)])


class InMemoryRowsDocProvider(DocumentProvider):
    """ Documents kept in memory.

    `rows` is a dict that maps a document id to a string or a dict
    (text under `text_key`).
    """

    def __init__(self, rows, content_to_sentences, text_key="text", docs_limit=None):
        assert(callable(content_to_sentences))
        assert(isinstance(rows, dict))
        assert(isinstance(docs_limit, int) or docs_limit is None)
        self.__sentences = content_to_sentences
        self.__rows = {}
        for count, (doc_id, value) in enumerate(rows.items()):
            if docs_limit is not None and count >= docs_limit:
                break
            self.__rows[str(doc_id)] = self.__text_of(value, text_key)

    @staticmethod
    def __text_of(value, text_key):
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            if text_key in value:
                return value[text_key]
            if "text" in value:
                return value["text"]
            raise KeyError("Row has no text field '{0}'".format(text_key))
        raise TypeError("Unsupported row type: {0}".format(type(value)))

    def by_id(self, doc_id):
        assert(isinstance(doc_id, str))
        text = self.__rows[doc_id]
        sentences = list(map(lambda sent: BaseDocumentSentence(sent), self.__sentences(text)))
        return Document(doc_id=doc_id, sentences=sentences)

    def iter_doc_ids(self):
        return iter(self.__rows)

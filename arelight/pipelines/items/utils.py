from arekit.common.docs.base import Document
from arekit.common.docs.sentence import BaseDocumentSentence


def input_to_docs(input_data, sentence_parser):
    """ input_data: list
        sentence_splitter: object
            how data is suppose to be separated onto sentences.
            str -> list(str)
    """
    assert(input_data is not None)

    docs = []

    for doc_id, contents in enumerate(input_data):
        # setup input data.
        sentences = sentence_parser(contents)
        sentences = list(map(lambda text: BaseDocumentSentence(text), sentences))
        # Documents.
        docs.append(Document(doc_id=doc_id, sentences=sentences))

    return docs

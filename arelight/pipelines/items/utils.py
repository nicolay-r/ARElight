from arekit.common.news.base import News
from arekit.common.news.sentence import BaseNewsSentence


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
        sentences = list(map(lambda text: BaseNewsSentence(text), sentences))
        # Documents.
        docs.append(News(doc_id=doc_id, sentences=sentences))

    return docs

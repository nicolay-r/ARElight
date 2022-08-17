from arekit.common.news.base import News
from arekit.common.news.sentence import BaseNewsSentence
from ru_sent_tokenize import ru_sent_tokenize


def input_to_docs(input_data):
    docs = []

    for doc_id, contents in enumerate(input_data):
        # setup input data.
        sentences = ru_sent_tokenize(contents)
        sentences = list(map(lambda text: BaseNewsSentence(text), sentences))
        # Parse text.
        doc = News(doc_id=doc_id, sentences=sentences)
        # Documents.
        docs.append(doc)

    return docs

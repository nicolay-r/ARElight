import unittest

from arekit.common.news.base import News
from arekit.common.news.parser import NewsParser
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem


class BertOntonotesPipelineItemTest(unittest.TestCase):

    text = "США пытается ввести санкции против Российской Федерацией"

    def test_pipeline(self):
        text_parser = BaseTextParser([
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(ner_model_cfg="ontonotes_mult")
        ])
        news = News(doc_id=0, sentences=[BaseNewsSentence(self.text)])
        parsed_news = NewsParser.parse(news=news, text_parser=text_parser)
        terms = parsed_news.iter_sentence_terms(sentence_index=0, return_id=False)

        for term in terms:
            print(term)


if __name__ == '__main__':
    unittest.main()

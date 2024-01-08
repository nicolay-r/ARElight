import unittest

from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.synonyms import iter_synonym_groups
from arelight.utils import IdAssigner

from utils_ner import test_ner


class TestDeepPavlovNERPipeline(unittest.TestCase):
    """ This unit test represent a tutorial on how
        AREkit might be applied towards data preparation for BERT
        model.
    """

    @staticmethod
    def iter_groups(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for data in iter_synonym_groups(file):
                yield data

    def test(self):

        texts = [
            """24 марта президент США Джо-Байден провел переговоры с
               лидерами стран Евросоюза в Брюсселе , вызвав внимание рынка и предположения о
               том, что Америке удалось уговорить ЕС совместно бойкотировать российские нефть
               и газ.  Европейский-Союз крайне зависим от России в плане поставок нефти и
               газа."""
        ]

        ner_ppl_items = [
            BasePipelineItem(src_func=lambda s: s.Text),
            DeepPavlovNERPipelineItem(src_func=lambda text: split_by_whitespaces(text),
                                      id_assigner=IdAssigner(),
                                      ner_model_name="ner_ontonotes_bert_mult")
        ]

        test_ner(texts=texts, ner_ppl_items=ner_ppl_items, collection_name="dp_ner")


if __name__ == '__main__':
    unittest.main()

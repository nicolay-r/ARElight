import unittest
from os.path import join, dirname, realpath

from arekit.common.entities.base import Entity
from arekit.common.text.enums import TermFormat
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem


class BertOntonotesPipelineItemTest(unittest.TestCase):
    """ Support text chunking.
    """

    def test_pipeline_item_rus(self):

        # Declaring text processing pipeline.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                                      ner_model_cfg="ontonotes_mult"),
        ])

        # Read file contents.
        text_filepath = join(dirname(realpath(__file__)), "../data/texts-inosmi-rus/e2.txt")
        with open(text_filepath, 'r') as f:
            text = f.read().rstrip()

        # Output parsed text.
        parsed_text = text_parser.run(text)
        for t in parsed_text.iter_terms(TermFormat.Raw):
            print("<{}> ({})".format(t.Value, t.Type) if isinstance(t, Entity) else t)

    def test_pipeline_item_eng_book(self):

        # Declaring text processing pipeline.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                                      ner_model_cfg="ontonotes_eng"),
        ])

        # Read file contents.
        text_filepath = join(dirname(realpath(__file__)), "../data/book-war-and-peace-test.txt")
        with open(text_filepath, 'r') as f:
            text = f.read().rstrip()

        # Output parsed text.
        parsed_text = text_parser.run(text)
        for t in parsed_text.iter_terms(TermFormat.Raw):
            print("<{}> ({})".format(t.Value, t.Type) if isinstance(t, Entity) else t)


if __name__ == '__main__':
    unittest.main()

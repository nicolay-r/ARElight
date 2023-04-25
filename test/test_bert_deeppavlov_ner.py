import unittest
from os.path import dirname, realpath, join

from arekit.common.entities.base import Entity
from arekit.common.text.enums import TermFormat
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.ner.deep_pavlov import DeepPavlovNER
from arelight.ner.obj_desc import NerObjectDescriptor
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem


class BertOntonotesTest(unittest.TestCase):

    def test_single_inference(self):
        """ Low level call of the NER.
            Note: Applicable only for a short input!
        """

        text = ".. При этом Москва неоднократно подчеркивала, что ее активность " \
               "на балтике является ответом именно на действия НАТО и эскалацию " \
               "враждебного подхода к Росcии вблизи ее восточных границ ..."

        ner = DeepPavlovNER(model_cfg="ontonotes_mult")
        tokens = text.split(' ')
        sequences = ner.extract(sequences=[tokens])

        print(len(sequences))
        for s_objs in sequences:
            for s_obj in s_objs:
                assert (isinstance(s_obj, NerObjectDescriptor))
                print("----")
                print(s_obj.ObjectType)
                print(s_obj.Position)
                print(s_obj.Length)
                print(tokens[s_obj.Position:s_obj.Position + s_obj.Length])

    def test_deeppavlov_ner_inference_ppl(self):
        """ Support text chunking.
        """

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


if __name__ == '__main__':
    unittest.main()

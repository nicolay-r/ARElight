import unittest
from arelight.ner.deep_pavlov import DeepPavlovNER
from arelight.ner.obj_desc import NerObjectDescriptor


class BertOntonotesTest(unittest.TestCase):

    text = ".. При этом Москва неоднократно подчеркивала, что ее активность " \
           "на балтике является ответом именно на действия НАТО и эскалацию " \
           "враждебного подхода к Росcии вблизи ее восточных границ ..."

    def test_single_inference(self):
        """ Low level call of the NER.
            Note: Applicable only for a short input!
        """

        ner = DeepPavlovNER(model_name="ner_ontonotes_bert_mult")
        tokens = self.text.split(' ')
        it = ner.extract(sequences=[tokens])

        for _, desc_list in it:
            for s_obj in desc_list:
                assert (isinstance(s_obj, NerObjectDescriptor))
                print("----")
                print(s_obj.ObjectType)
                print(s_obj.Position)
                print(s_obj.Length)
                print(tokens[s_obj.Position:s_obj.Position + s_obj.Length])


if __name__ == '__main__':
    unittest.main()

import json
import unittest
from os.path import dirname, realpath, join

from arelight.demo.infer_bert_rus import demo_infer_texts_bert


class TestDemo(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    ORIGIN_DATA_DIR = join(current_dir, "../data")

    @staticmethod
    def __prepare_template(data, text, template_filepath):
        assert(isinstance(data, dict))

        with open(template_filepath, "r") as template_file:
            template_local = template_file.read()

        template_local = template_local.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data', '')))
        template_local = template_local.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data', '')))
        template_local = template_local.replace("$____TEXT____", text)

        return template_local

    def test_demo_rus_bert(self):
        text = "США вводит сакнции против РФ"
        contents = demo_infer_texts_bert(
            text=text,
            output_dir=".",
            state_name="ra-20-srubert-large-neut-nli-pretrained-3l",
            finetuned_state_name="ra-20-srubert-large-neut-nli-pretrained-3l-finetuned",
            model_dir=join(TestDemo.ORIGIN_DATA_DIR, "models"),
            synonyms_filepath=join(TestDemo.ORIGIN_DATA_DIR, "synonyms.txt"))

        template = TestDemo.__prepare_template(
            data=contents, text=text,
            template_filepath=join(TestDemo.ORIGIN_DATA_DIR, "brat_template.html"))

        print(template)


if __name__ == '__main__':
    unittest.main()

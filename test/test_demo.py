import json
import unittest
from os.path import dirname, realpath, join

from arelight.demo.infer_bert_rus import demo_infer_texts_bert_pipeline


class TestDemo(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    TEST_DATA_DIR = join(current_dir, "data")
    ORIGIN_DATA_DIR = join(current_dir, "../data")

    @staticmethod
    def __prepare_template(data, text, template_filepath, brat_url):
        assert(isinstance(brat_url, str))
        assert(isinstance(data, dict))

        with open(template_filepath, "r") as template_file:
            template_local = template_file.read()

        template_local = template_local.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data', '')))
        template_local = template_local.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data', '')))
        template_local = template_local.replace("$____BRAT_URL____", brat_url)
        template_local = template_local.replace("$____TEXT____", text)

        return template_local

    def test_demo_rus_bert(self):
        text = "США вводит санкции против РФ"

        model_dir = join(self.ORIGIN_DATA_DIR, "models")
        state_name = "ra-20-srubert-large-neut-nli-pretrained-3l"
        finetuned_state_name = "ra-20-srubert-large-neut-nli-pretrained-3l-finetuned"

        ppl = demo_infer_texts_bert_pipeline(
            texts_count=1,
            output_dir=".",
            bert_vocab_path=join(model_dir, state_name, "vocab.txt"),
            bert_config_path=join(model_dir, state_name, "bert_config.json"),
            bert_finetuned_ckpt_path=join(model_dir, finetuned_state_name, state_name),
            synonyms_filepath=join(TestDemo.ORIGIN_DATA_DIR, "synonyms.txt"))

        contents = ppl.run([text])

        template = TestDemo.__prepare_template(
            data=contents, text=text,
            template_filepath=join(TestDemo.ORIGIN_DATA_DIR, "brat_template.html"),
            brat_url="http://localhost:8001/")

        with open(join(self.TEST_DATA_DIR, "demo-rus-bert-output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()

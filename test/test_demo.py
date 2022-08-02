import json
import unittest
from os.path import dirname, realpath, join

from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter

from arelight.demo.infer_bert_rus import demo_infer_texts_bert_pipeline
from arelight.demo.infer_nn_rus import demo_infer_texts_tensorflow_nn_pipeline
from arelight.demo.labels.formatter import ExperimentRuSentiFramesLabelsFormatter
from arelight.demo.labels.scalers import ThreeLabelScaler
from examples.args import const


class TestDemo(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    TEST_DATA_DIR = join(current_dir, "data")
    ORIGIN_DATA_DIR = join(current_dir, "../data")

    text = "США вводит санкции против РФ"

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

    def test_demo_rus_nn(self):
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=ExperimentRuSentiFramesLabelsFormatter())

        ppl = demo_infer_texts_tensorflow_nn_pipeline(
            texts_count=1,
            output_dir=".",
            model_name=ModelNames.PCNN,
            entity_fmt=StringEntitiesSimpleFormatter(),
            synonyms_filepath=join(TestDemo.ORIGIN_DATA_DIR, "synonyms.txt"),
            model_load_dir=const.NEURAL_NETWORKS_TARGET_DIR,
            model_input_type=ModelInputType.SingleInstance,
            frames_collection=frames_collection)

        contents = ppl.run([self.text])

        template = TestDemo.__prepare_template(
            data=contents, text=self.text,
            template_filepath=join(TestDemo.ORIGIN_DATA_DIR, "brat_template.html"),
            brat_url="http://localhost:8001/")

        with open(join(self.TEST_DATA_DIR, "demo-rus-nn-output.html"), "w") as output:
            output.write(template)

    def test_demo_rus_bert(self):

        model_dir = join(self.ORIGIN_DATA_DIR, "models")
        state_name = "ra-20-srubert-large-neut-nli-pretrained-3l"
        finetuned_state_name = "ra-20-srubert-large-neut-nli-pretrained-3l-finetuned"

        ppl = demo_infer_texts_bert_pipeline(
            texts_count=1,
            output_dir=".",
            labels_scaler=ThreeLabelScaler(),
            entity_fmt=SharpPrefixedEntitiesSimpleFormatter(),
            bert_vocab_path=join(model_dir, state_name, "vocab.txt"),
            bert_config_path=join(model_dir, state_name, "bert_config.json"),
            bert_finetuned_ckpt_path=join(model_dir, finetuned_state_name, state_name),
            synonyms_filepath=join(TestDemo.ORIGIN_DATA_DIR, "synonyms.txt"))

        contents = ppl.run([self.text])

        template = TestDemo.__prepare_template(
            data=contents, text=self.text,
            template_filepath=join(TestDemo.ORIGIN_DATA_DIR, "brat_template.html"),
            brat_url="http://localhost:8001/")

        with open(join(self.TEST_DATA_DIR, "demo-rus-bert-output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()

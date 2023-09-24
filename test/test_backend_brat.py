import json
import unittest
import utils
from os.path import join

from arelight.backend.brat.converter import BratBackend
from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler


class TestBratEmbedding(unittest.TestCase):

    def __to_html(self, template_filepath, contents, brat_url):
        # Loading template file.
        with open(template_filepath, "r") as templateFile:
            template = templateFile.read()

        # Replace template placeholders.
        template = template.replace("$____COL_DATA_SEM____", json.dumps(contents["coll_data"]))
        template = template.replace("$____DOC_DATA_SEM____", json.dumps(contents["doc_data"]))
        template = template.replace("$____BRAT_URL____", brat_url)
        template = template.replace("$____TEXT____", contents["text"])

        return template

    def __create_template(self, infer_predict_filepath, samples_data_filepath, labels_scaler, docs_range=(0, 5)):
        brat_be = BratBackend()

        contents = brat_be.to_data(
            infer_predict_filepath=infer_predict_filepath,
            samples_data_filepath=samples_data_filepath,
            label_to_rel={str(labels_scaler.label_to_uint(PositiveLabel())): "POS",
                          str(labels_scaler.label_to_uint(NegativeLabel())): "NEG"},
            obj_color_types={"ORG": '#7fa2ff',
                             "GPE": "#7fa200",
                             "PER": "#7f00ff",
                             "LOC": "#5f00aa",
                             "PERSON": "#7f00ff",
                             "GEOPOLIT": "#7fa200",
                             "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN",
                             "NEG": "RED"},
            docs_range=docs_range)

        return self.__to_html(template_filepath=join(utils.TEST_DATA_DIR, "brat_template.html"),
                              contents=contents,
                              brat_url="http://localhost:8001/")

    def test(self):
        template = self.__create_template(
            samples_data_filepath=join(utils.TEST_DATA_DIR, "brat-backend-samples-test.csv"),
            infer_predict_filepath=join(utils.TEST_OUT_DIR, "brat-backend-predict.tsv.gz"),
            labels_scaler=ThreeLabelScaler())

        with open(join(utils.TEST_OUT_DIR, "brat-backend-output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()

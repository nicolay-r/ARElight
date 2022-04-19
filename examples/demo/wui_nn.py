#!/root/.pyenv/shims/python
# -*- coding: utf8 -*-

import cgitb
import cgi
import json
import sys
from os.path import join, basename

from arekit.contrib.experiment_rusentrel.labels.formatters.rusentiframes import ExperimentRuSentiFramesLabelsFormatter
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions

from arelight.demo.infer_nn_rus import demo_infer_texts_tensorflow_nn_pipeline


bratUrl = '/brat/'


def cgi_output(data):
    sys.stdout.buffer.write("Content-type: text/html\n\n".encode('utf8'))
    sys.stdout.buffer.write(data.encode('utf8'))


def prepare_template(data, text, bratUrl, model_name):
    assert(isinstance(model_name, ModelNames))
    assert(isinstance(data, dict))

    with open("index-template.html", "r") as template_file:
        template_local = template_file.read()

    template_local = template_local.replace("$____MODEL_NAME____", model_name.value)
    template_local = template_local.replace("$____MODEL_DESCRIPTION____", "(RuSentRel finetuned)")

    template_local = template_local.replace("$____SCRIPT_NAME____", basename(__file__))

    template_local = template_local.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data', '')))
    template_local = template_local.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data', '')))

    template_local = template_local.replace("$____TEXT____", text)
    template_local = template_local.replace("$____BRAT_URL____", bratUrl)

    return template_local


cgitb.enable(display=0, logdir="/")
inputData = cgi.FieldStorage()
text = inputData.getfirst("text")
model_name = ModelNames.PCNN

if not text:
    template = prepare_template(data={},
                                text="США вводит санкции против РФ",
                                bratUrl=bratUrl,
                                model_name=model_name)
    cgi_output(template)
    exit(0)


data_dir = "/arelight/data"
state_name = "ra-20-srubert-large-neut-nli-pretrained-3l"
finetuned_state_name = "ra-20-srubert-large-neut-nli-pretrained-3l-finetuned"

frames_collection = RuSentiFramesCollection.read_collection(
    version=RuSentiFramesVersions.V20,
    labels_fmt=ExperimentRuSentiFramesLabelsFormatter())

ppl = demo_infer_texts_tensorflow_nn_pipeline(
    texts_count=1,
    output_dir=".",
    model_name=model_name,
    model_input_type=ModelInputType.SingleInstance,
    synonyms_filepath=join(data_dir, "synonyms.txt"),
    model_load_dir=join(data_dir, "models"),
    embedding_filepath=join(data_dir, "news_mystem_skipgram_1000_20_2015.bin.gz"),
    frames_collection=frames_collection)

brat_json = ppl.run([text.strip()])

template = prepare_template(data=brat_json,
                            text=text,
                            bratUrl=bratUrl,
                            model_name=model_name)

cgi_output(template)

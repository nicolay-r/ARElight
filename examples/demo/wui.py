#!/root/.pyenv/shims/python
# -*- coding: utf8 -*-

import cgitb
import cgi
import json
import sys
import os
from os.path import join

from arelight.demo.infer_bert_rus import demo_infer_texts_bert_pipeline

ip_address = os.environ['IP_ADDRESS']
bratUrl = '/brat/'


def cgi_output(data):
    sys.stdout.buffer.write("Content-type: text/html\n\n".encode('utf8'))
    sys.stdout.buffer.write(data.encode('utf8'))


def prepare_template(data, text, bratUrl):
    assert (isinstance(data, dict))

    with open("index.tmpl", "r") as template_file:
        template_local = template_file.read()

    template_local = template_local.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data', '')))
    template_local = template_local.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data', '')))

    template_local = template_local.replace("$____TEXT____", text)
    template_local = template_local.replace("$____BRAT_URL____", bratUrl)

    return template_local


cgitb.enable(display=0, logdir="/")
inputData = cgi.FieldStorage()
text = inputData.getfirst("text")

if not text:
    template = prepare_template(data={},
                                text="США вводит санкции против РФ",
                                bratUrl=bratUrl)
    cgi_output(template)
    exit(0)


model_dir = "/arelight/data/models"
state_name = "ra-20-srubert-large-neut-nli-pretrained-3l"
finetuned_state_name = "ra-20-srubert-large-neut-nli-pretrained-3l-finetuned"

ppl = demo_infer_texts_bert_pipeline(
    texts_count=1,
    output_dir=".",
    bert_config_path=join(model_dir, state_name, "bert_config.json"),
    bert_vocab_path=join(model_dir, state_name, "vocab.txt"),
    bert_finetuned_ckpt_path=join(model_dir, finetuned_state_name, state_name),
    synonyms_filepath="/arelight/data/synonyms.txt")

brat_json = ppl.run([text.strip()])

template = prepare_template(brat_json, text, bratUrl)
cgi_output(template)

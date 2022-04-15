#!/root/.pyenv/shims/python
# -*- coding: utf8 -*-

import cgitb
import cgi
import json
import sys
import os

from examples.demo.infer_texts_bert import infer_texts_bert

ip_address = os.environ['IP_ADDRESS']
morph_port = int(os.environ['MORPH_PORT'])
syntax_port = int(os.environ['SYNTAX_PORT'])
sem_port = int(os.environ['SEM_PORT'])

bratUrl = '/demo/brat/'


def cgi_output(data):
    sys.stdout.buffer.write("Content-type: text/html\n\n".encode('utf8'))
    sys.stdout.buffer.write(data.encode('utf8'))


def prepare_template(data, text, bratUrl):
    template_file = open("index.tmpl", "r")
    template_local = template_file.read()

    template_local = template_local.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data_sem', '')))
    template_local = template_local.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data_sem', '')))

    template_local = template_local.replace("$____TEXT____", text)
    template_local = template_local.replace("$____BRAT_URL____", bratUrl)

    return template_local


cgitb.enable()
inputData = cgi.FieldStorage()
text = inputData.getfirst("text")

if not text:
    template = prepare_template(data={}, text="", bratUrl=bratUrl)
    cgi_output(template)
    exit(0)

brat_json = infer_texts_bert(text="... сша пытается ввести санкции против россии ...")
template = prepare_template(brat_json, text, bratUrl)
cgi_output(template)

#!/root/.pyenv/shims/python
# -*- coding: utf8 -*-

import cgitb
import cgi
import json
import sys
from os.path import join, basename

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arelight.demo.infer_bert_rus import demo_infer_texts_bert_pipeline
from arelight.demo.labels.scalers import ThreeLabelScaler
from arelight.demo.utils import read_synonyms_collection
from arelight.exp.doc_ops import InMemoryDocOperations
from arelight.pipelines.annot_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.utils import input_to_docs
from arelight.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem

bratUrl = '/brat/'


def cgi_output(data):
    sys.stdout.buffer.write("Content-type: text/html\n\n".encode('utf8'))
    sys.stdout.buffer.write(data.encode('utf8'))


def prepare_template(data, text, bratUrl):
    assert (isinstance(data, dict))

    with open("index-template.html", "r") as template_file:
        template_local = template_file.read()

    template_local = template_local.replace("$____MODEL_NAME____",
                                            "SentRuBERT")

    template_local = template_local.replace("$____MODEL_DESCRIPTION____",
                                            "(ra-20-srubert-large-neut-nli-pretrained-3l-finetuned)")

    template_local = template_local.replace("$____SCRIPT_NAME____", basename(__file__))

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

demo_pipeline = demo_infer_texts_bert_pipeline(
    texts_count=1,
    output_dir=".",
    entity_fmt=SharpPrefixedEntitiesSimpleFormatter(),
    bert_config_path=join(model_dir, state_name, "bert_config.json"),
    bert_vocab_path=join(model_dir, state_name, "vocab.txt"),
    bert_finetuned_ckpt_path=join(model_dir, finetuned_state_name, state_name),
    synonyms_filepath="/arelight/data/synonyms.txt",
    labels_scaler=ThreeLabelScaler())

synonyms = read_synonyms_collection(synonyms_filepath="/arelight/data/synonyms.txt",
                                    stemmer=MystemWrapper())

text_parser = BaseTextParser(pipeline=[
    TermsSplitterParser(),
    BertOntonotesNERPipelineItem(lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"]),
    EntitiesGroupingPipelineItem(
        lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
            synonyms=synonyms, value=value))
])

# Declare a single document with `0` id and contents.
single_doc = (0, text.strip())
doc_ops = InMemoryDocOperations(docs=input_to_docs(single_doc))

data_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                   terms_per_context=50,
                                                   dist_in_terms_bound=50,
                                                   doc_ops=doc_ops,
                                                   text_parser=text_parser)

no_folding = NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])

brat_json = demo_pipeline.run(None, {
    "data_type_pipelines": {DataType.Test: data_pipeline},
    "data_folding": no_folding
})

template = prepare_template(brat_json, text, bratUrl)
cgi_output(template)

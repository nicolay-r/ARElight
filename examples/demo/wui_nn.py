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
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.utils.pipelines.items.text.frames import FrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arelight.demo.infer_nn_rus import demo_infer_texts_tensorflow_nn_pipeline
from arelight.demo.labels.formatter import ExperimentRuSentiFramesLabelsFormatter
from arelight.demo.utils import read_synonyms_collection
from arelight.exp.doc_ops import InMemoryDocOperations
from arelight.network.nn.common import create_and_fill_variant_collection
from arelight.pipelines.annot_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.utils import input_to_docs
from arelight.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem

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

demo_pipeline = demo_infer_texts_tensorflow_nn_pipeline(
    texts_count=1,
    output_dir=".",
    model_name=model_name,
    model_input_type=ModelInputType.SingleInstance,
    entity_fmt=StringEntitiesSimpleFormatter(),
    model_load_dir=join(data_dir, "models"),
    frames_collection=frames_collection)

stemmer = MystemWrapper()
synonyms = read_synonyms_collection(synonyms_filepath="/arelight/data/synonyms.txt", stemmer=stemmer)

demo_pipeline.append(BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/"))

# Declare a single document with `0` id and contents.
single_doc = (0, text.strip())
doc_ops = InMemoryDocOperations(docs=input_to_docs(single_doc))

# Initialize text parser with the related dependencies.
frame_variants_collection = create_and_fill_variant_collection(frames_collection)
text_parser = BaseTextParser(pipeline=[
    TermsSplitterParser(),
    BertOntonotesNERPipelineItem(lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"]),
    EntitiesGroupingPipelineItem(
        lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
            synonyms, value)),
    DefaultTextTokenizer(keep_tokens=True),
    FrameVariantsParser(frame_variants=frame_variants_collection),
    LemmasBasedFrameVariantsParser(save_lemmas=False,
                                   stemmer=stemmer,
                                   frame_variants=frame_variants_collection),
    FrameVariantsSentimentNegation()])

data_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                   dist_in_terms_bound=50,
                                                   terms_per_context=50,
                                                   doc_ops=doc_ops,
                                                   text_parser=text_parser,
                                                   dist_in_sentences=0)

no_folding = NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])

brat_json = demo_pipeline.run([text.strip()], {
    "data_type_pipelines": {DataType.Test: data_pipeline},
    "data_folding": no_folding
})

template = prepare_template(data=brat_json, text=text, bratUrl=bratUrl, model_name=model_name)
cgi_output(template)

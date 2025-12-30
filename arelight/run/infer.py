import argparse
from os.path import join, dirname, basename

from arekit.common.data import const
from arekit.common.data.const import ID
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection
from bulk_ner.src.pipeline.item.ner import NERPipelineItem
from bulk_ner.src.utils import IdAssigner

from arelight.api import create_inference_pipeline
from arelight.arekit.indexed_entity import IndexedEntity
from arelight.arekit.samples_io import CustomSamplesIO
from arelight.arekit.utils_translator import string_terms_to_list
from arelight.const import BULK_CHAIN, D3JS_GRAPHS
from arelight.data.writers.sqlite_native import SQliteWriter
from arelight.doc_provider import CachedFilesDocProvider
from arelight.entity import HighligtedEntitiesFormatter
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_llm import demo_infer_texts_llm_pipeline
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.pipelines.demo.result import PipelineResult
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.predict.writer_sqlite3 import SQLite3PredictWriter
from arelight.readers.csv_pd import PandasCsvReader
from arelight.readers.sqlite import SQliteReader
from arelight.run.utils import merge_dictionaries, iter_group_values, create_sentence_parser, iter_content, NER_TYPES
from arelight.run.utils_logger import setup_custom_logger, TqdmToLogger
from arelight.samplers.cropped import create_prompted_sample_provider
from arelight.stemmers.ru_mystem import MystemWrapper
from arelight.third_party.dp_130 import DeepPavlovNER
from arelight.third_party.gt_310a import GoogleTranslateModel
from arelight.utils import flatten

from bulk_translate.src.pipeline.translator import MLTextTranslatorPipelineItem


def create_infer_parser():

    parser = argparse.ArgumentParser(description="Text inference example")

    # TODO. refactor using the concept of declared class of fields.
    # TODO. From fields we can create arguments.

    # Providing arguments.
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=10, nargs='?')
    parser.add_argument('--from-files', dest='from_files', type=str, default=None, nargs='+')
    parser.add_argument('--csv-sep', dest='csv_sep', type=str, default=',', nargs='?', choices=["\t", ',', ';'])
    parser.add_argument('--csv-column', dest='csv_column', type=str, default='text', nargs='?')
    parser.add_argument('--collection-name', dest='collection_name', type=str, default=None, nargs='+')
    # AREkit parameters.
    parser.add_argument('--sampling-framework', dest='sampling_framework', type=str, choices=[None, "arekit"], default=None)
    parser.add_argument('--terms-per-context', dest='terms_per_context', type=int, default=50, nargs='?', help='The max possible length of an input context in terms')
    parser.add_argument('--sentence-parser', dest='sentence_parser', type=str, default="nltk:english", choices=["nltk:english", "nltk:russian"])
    parser.add_argument('--synonyms-filepath', dest='synonyms_filepath', type=str, default=None, help="List of synonyms provided in lines of the source text file.")
    parser.add_argument('--stemmer', dest='stemmer', type=str, default=None, choices=[None, "mystem"])
    # NER part.
    parser.add_argument("--ner-framework", dest="ner_framework", type=str, choices=["deeppavlov"], default="deeppavlov")
    parser.add_argument('--ner-model-name', dest='ner_model_name', type=str, default=None, choices=["ner_ontonotes_bert", "ner_ontonotes_bert_mult"])
    parser.add_argument('--ner-types', dest='ner_types', type=str, default= "|".join(NER_TYPES), help="Filters specific NER types; provide with `|` separator")
    # Translation parameters.
    parser.add_argument('--translate-framework', dest='translate_framework', type=str, default=None, choices=[None, "googletrans"])
    parser.add_argument('--translate-entity', dest='translate_entity', type=str, default=None, choices=[None, "auto:ru"])
    parser.add_argument('--translate-text', dest='translate_text', type=str, default=None, choices=[None, "auto:ru"])
    # Inference parameters.
    parser.add_argument("--inference-framework", dest="inference_framework", type=str, default=BULK_CHAIN, choices=[BULK_CHAIN])
    parser.add_argument("--inference-api", dest="inference_api", type=str, default=None)
    parser.add_argument('--inference-filename', dest="inference_filename", type=str, default=None)
    parser.add_argument('--inference-model-name', dest="inference_model_name", type=str, default=None)
    parser.add_argument('--inference-writer', dest="inference_writer", type=str, default="sqlite3", choices=["sqlite3", "tsv"])
    # Common parameters.
    parser.add_argument("--docs-limit", dest="docs_limit", type=int, default=None)
    parser.add_argument("--labels-fmt", dest="labels_fmt", default="u:0,p:1,n:2", type=str)
    parser.add_argument("--device-type", dest="device_type", type=str, default="cpu", help="Device type applicable for launching machine learning models", choices=['cpu', 'cuda'])
    # Backend parameters.
    parser.add_argument("--backend", dest="backend", type=str, default=None, choices=[None, D3JS_GRAPHS])
    parser.add_argument("--label-names", dest="d3js_label_names", type=str, default="p:pos,n:neg,u:neu")
    # Logging parameters.
    parser.add_argument('--log-file', dest="log_file", default=None, type=str)
    parser.add_argument('-o', '--output-template', dest='output_template', type=str, default="output", nargs='?')

    return parser


def setup_collection_name(value):
    # Considering Predefined name if the latter has been declared.
    if value is not None:
        return value
    # Use the name of the file.
    if args.from_files is not None:
        return basename(args.from_files[0]) if len(args.from_files) == 1 else "from-many-files"
    if args.from_dataframe is not None:
        return basename(args.from_dataframe[0])

    return "samples"


if __name__ == '__main__':


    # Completing list of arguments.
    parser = create_infer_parser()

    # Parsing arguments.
    args = parser.parse_args()

    # Creating pipeline.
    pipeline, settings = create_inference_pipeline(
        args=args, 
        collection_name=setup_collection_name(args.collection_name)
    )

    # Launch application.
    BasePipelineLauncher.run(pipeline=pipeline, pipeline_ctx=PipelineResult(merge_dictionaries(settings)),
                             has_input=False)

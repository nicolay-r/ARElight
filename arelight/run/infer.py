import argparse
from ntpath import dirname
from os.path import basename, join

from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.contrib.prompt.sample import BaseSingleTextProvider
from bulk_chain.core.utils import dynamic_init

from arelight.api import create_inference_pipeline
from arelight.const import BULK_CHAIN, D3JS_GRAPHS
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter
from arelight.pipelines.result import PipelineResult
from arelight.run.utils import merge_dictionaries, NER_TYPES
from arelight.run.utils_logger import TqdmToLogger, setup_custom_logger

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
    # NER part.
    parser.add_argument('--ner-provider', dest='ner_provider', type=str, default=None)
    parser.add_argument('--ner-model-name', dest='ner_model_name', type=str, default=None)
    parser.add_argument('--ner-types', dest='ner_types', type=str, default= "|".join(NER_TYPES), help="Filters specific NER types; provide with `|` separator")
    # Translation parameters.
    parser.add_argument('--translate-provider', dest='translate_provider', type=str, default=None)
    parser.add_argument('--translate-text', dest='translate_text', type=str, default=None)
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


def class_to_int(text):
    if 'positive' in text.lower():
        return 1
    elif 'negative' in text.lower():
        return -1 
    return 0 


if __name__ == '__main__':

    # Completing list of arguments.
    parser = create_infer_parser()

    # Parsing arguments.
    args = parser.parse_args()

    # Setup logger
    logger = setup_custom_logger(name="arelight", filepath=args.log_file)
    tqdm_log_out = TqdmToLogger(logger) if args.log_file is not None else None

    # Other parameters.
    predict_table_name = "bulk_chain"
    collection_name = setup_collection_name(args.collection_name)
    output_dir = dirname(args.output_template) if dirname(args.output_template) != "" else args.output_template
    collection_target_func = lambda data_type: join(output_dir, "-".join([collection_name, data_type.name.lower()]))

    # Init NER model.
    ner_model_type = dynamic_init(class_filepath=args.ner_provider)

    # Init Translator model.
    translate_model = dynamic_init(class_filepath=args.translate_provider) if args.translate_provider is not None else None

    # Creating pipeline.
    pipeline, settings = create_inference_pipeline(
        args=args,
        files_iter=args.from_files,
        predict_table_name=predict_table_name,
        collection_target_func=collection_target_func,
        translator_args={
            "model": translate_model() if translate_model is not None else None,
            "src": args.translate_text.split(':')[0] if args.translate_text is not None else None,
            "dest": args.translate_text.split(':')[1] if args.translate_text is not None else None,
        },
        ner_args={
            "model": ner_model_type(model=args.ner_model_name),
            "obj_filter": None if args.ner_types is None else lambda s_obj: s_obj.ObjectType in args.ner_types,
            "chunk_limit": 128
        },
        inference_args={
            "model": args.inference_model_name,
            "class_name": args.inference_filename,
            "api_key": args.inference_api,
            "task_kwargs": {
                "schema": [{
                    "prompt": f"Given text: {{{BaseSingleTextProvider.TEXT_A}}}" +
                              f"TASK: Classify sentiment attitude of [SUBJECT] to [OBJECT]: "
                              f"positive, "
                              f"negative, "
                              f"neutral.",
                    "out": "response"
                }],
                "classify_func": lambda row: class_to_int(row['response']),
            }
        },
        tqdm_log_out=tqdm_log_out
    )

    # TODO. This is temporary for supporting legacy backend settings.
    if args.backend == "d3js_graphs":
        labels_fmt = {a: v for a, v in map(lambda item: item.split(":"), args.d3js_label_names.split(','))}
        settings.append({
            "labels_formatter": CustomLabelsFormatter(**labels_fmt),
            "d3js_collection_name": collection_name,
            "d3js_collection_description": collection_name,
            "d3js_graph_output_dir": output_dir
        })

    # Launch application.
    BasePipelineLauncher.run(pipeline=pipeline, pipeline_ctx=PipelineResult(merge_dictionaries(settings)),
                             has_input=False)

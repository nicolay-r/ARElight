import argparse
from importlib.util import module_from_spec, spec_from_file_location
from ntpath import dirname
from os.path import basename, dirname as path_dirname, join

from dotenv import load_dotenv

from arekit.common.pipeline.base import BasePipelineLauncher
from bulk_chain.core.utils import dynamic_init

from arelight.api import create_inference_pipeline
from arelight.const import BULK_CHAIN
from arelight.pipelines.result import PipelineResult
from arelight.run.utils import iter_content, merge_dictionaries, NER_TYPES
from arelight.run.utils_logger import TqdmToLogger, setup_custom_logger
from arelight.utils import get_event_loop


_LABEL_NAMES = {"u": "neutral", "p": "positive", "n": "negative"}


def attitude_task_kwargs(labels_fmt):
    """ Prompt for this command-line attitude example.

    The model replies with the integer declared in `labels_fmt`.
    """
    from arekit.common.data.input.providers.text.single import BaseSingleTextProvider

    choices = ", ".join(
        f"{value} for {_LABEL_NAMES[name]}"
        for name, value in (item.split(":") for item in labels_fmt.split(","))
    )
    prompt = (
        f"Given text: {{{BaseSingleTextProvider.TEXT_A}}}"
        "TASK: Classify sentiment attitude of [SUBJECT] to [OBJECT]. "
        f"Reply with a single integer: {choices}."
    )
    return {
        "schema": [{"prompt": prompt, "out": "response"}],
        "labels_fmt": labels_fmt,
    }


def create_sentence_parser(framework, language):
    """ Build a sentence splitter. NLTK lives in `demo/nltk.py`, outside the library. """
    if framework == "linesplit":
        return lambda text: [t.strip() for t in text.split('\n')]
    if framework == "nltk":
        demo_nltk = join(path_dirname(path_dirname(path_dirname(__file__))), "demo", "nltk.py")
        spec = spec_from_file_location("arelight_demo_nltk", demo_nltk)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.tokenize(language)
    raise Exception("Framework `{}` is not supported".format(framework))


def iter_rows_from_files(files_iter, csv_sep=",", csv_column="text", content_iter=None):

    def _default_content(filepath):
        return iter_content(filepath=filepath, csv_column=csv_column, csv_delimiter=csv_sep)

    read_content = content_iter if content_iter is not None else _default_content
    for filepath in files_iter or []:
        for index, content in enumerate(read_content(filepath)):
            yield {"id": ":".join([str(filepath), str(index)]), "text": content, "filename": filepath}


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
    parser.add_argument('--inference-filename', dest="inference_filename", type=str, default=None)
    parser.add_argument('--inference-model-name', dest="inference_model_name", type=str, default=None)
    parser.add_argument('--inference-writer', dest="inference_writer", type=str, default="sqlite3", choices=["sqlite3", "tsv"])
    # Common parameters.
    parser.add_argument("--docs-limit", dest="docs_limit", type=int, default=None)
    parser.add_argument("--labels-fmt", dest="labels_fmt", default="u:0,p:1,n:2", type=str)
    parser.add_argument("--device-type", dest="device_type", type=str, default="cpu", help="Device type applicable for launching machine learning models", choices=['cpu', 'cuda'])
    # Logging parameters.
    parser.add_argument('--log-file', dest="log_file", default=None, type=str)
    parser.add_argument('-o', '--output-template', dest='output_template', type=str, default="output", nargs='?')

    return parser


def setup_collection_name(value):
    # Considering Predefined name if the latter has been declared.
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else "-".join(value)
    if value is not None:
        return value
    # Use the name of the file.
    if args.from_files:
        return basename(args.from_files[0]) if len(args.from_files) == 1 else "from-many-files"

    return "samples" 


if __name__ == '__main__':

    load_dotenv()

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

    ner_types = None if args.ner_types is None else set(args.ner_types.split("|"))
    event_loop = get_event_loop()
    framework, language = args.sentence_parser.split(":")

    translator_args = None
    if translate_model is not None:
        src, dest = (None, None) if args.translate_text is None else args.translate_text.split(":")
        translator_args = {
            "model": translate_model(),
            "src": src,
            "dest": dest,
        }

    # Creating pipeline.
    pipeline, settings = create_inference_pipeline(
        rows_iter=iter_rows_from_files(
            files_iter=args.from_files,
            csv_sep=args.csv_sep,
            csv_column=args.csv_column),
        sampling_args={
            "terms_per_context": args.terms_per_context,
            "docs_limit": args.docs_limit,
            "synonyms_filepath": args.synonyms_filepath,
        },
        predict_table_name=predict_table_name,
        collection_target_func=collection_target_func,
        translator_args=translator_args,
        ner_args={
            "model": ner_model_type(model=args.ner_model_name),
            "obj_filter": None if ner_types is None else lambda s_obj: s_obj.ObjectType in ner_types,
            "chunk_limit": 128
        },
        inference_args={
            "model_name": args.inference_model_name,
            "class_name": args.inference_filename,
            "task_kwargs": attitude_task_kwargs(args.labels_fmt),
        },
        writer_args={
            "writer": args.inference_writer,
        },
        tqdm_log_out=tqdm_log_out,
        batch_size=args.batch_size,
        event_loop=event_loop,
        sentence_parser=create_sentence_parser(framework=framework, language=language),
    )

    # Launch application.
    BasePipelineLauncher.run(pipeline=pipeline, pipeline_ctx=PipelineResult(merge_dictionaries(settings)),
                             has_input=False)

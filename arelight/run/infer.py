import argparse
from os.path import join, dirname, basename

from arekit.common.data import const
from arekit.common.data.const import S_IND, T_IND, ID
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.readers.sqlite import SQliteReader
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.sqlite_native import SQliteWriter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.pipelines.items.text.translator import MLTextTranslatorPipelineItem
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection
from arekit.contrib.utils.synonyms.stemmer_based import StemmerBasedSynonymCollection

from arelight.arekit.samples_io import CustomSamplesIO
from arelight.doc_provider import CachedFilesDocProvider
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.pipelines.demo.result import PipelineResult
from arelight.pipelines.items.entities_default import TextEntitiesParser
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.pipelines.items.entities_ner_transformers import TransformersNERPipelineItem
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.predict.writer_sqlite3 import SQLite3PredictWriter
from arelight.run.utils import merge_dictionaries, iter_group_values, create_sentence_parser, \
    create_translate_model, is_port_number, iter_content, OPENNRE_CHECKPOINTS
from arelight.run.utils_logger import setup_custom_logger, TqdmToLogger
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import SampleFormattersService
from arelight.utils import IdAssigner


def create_infer_parser():

    parser = argparse.ArgumentParser(description="Text inference example")

    # Providing arguments.
    parser.add_argument('--from-files', dest='from_files', type=str, default=None, nargs='+')
    parser.add_argument('--csv-sep', dest='csv_sep', type=str, default=',', nargs='?')
    parser.add_argument('--csv-column', dest='csv_column', type=str, default='text', nargs='?')
    parser.add_argument('--collection-name', dest='collection_name', type=str, default=None, nargs='+')
    parser.add_argument('--terms-per-context', dest='terms_per_context', type=int, default=50, nargs='?', help='The max possible length of an input context in terms')
    parser.add_argument('--sentence-parser', dest='sentence_parser', type=str, default="nltk:english")
    parser.add_argument('--synonyms-filepath', dest='synonyms_filepath', type=str, default=None, help="List of synonyms provided in lines of the source text file.")
    parser.add_argument('--stemmer', dest='stemmer', type=str, default=None, choices=[None, "mystem"])
    parser.add_argument('--sampling-framework', dest='sampling_framework', type=str, choices=[None, "arekit"], default=None)
    parser.add_argument("--ner-framework", dest="ner_framework", type=str, choices=[None, "deeppavlov", "transformers"], default="deeppavlov")
    parser.add_argument('--ner-model-name', dest='ner_model_name', type=str, default=None)
    parser.add_argument('--ner-types', dest='ner_types', type=str, default="ORG|PERSON|LOC|GPE", help="Filters specific NER types; provide with `|` separator")
    parser.add_argument('--inference-writer', dest="inference_writer", type=str, default="sqlite3", choices=["sqlite3", "tsv"])
    parser.add_argument('--translate-framework', dest='translate_framework', type=str, default=None, choices=[None, "googletrans"])
    parser.add_argument('--translate-entity', dest='translate_entity', type=str, default=None)
    parser.add_argument('--translate-text', dest='translate_text', type=str, default=None)
    parser.add_argument("--docs-limit", dest="docs_limit", type=int, default=None)
    parser.add_argument('--text-b-type', dest='text_b_type', type=str, default=None, choices=list(SampleFormattersService.iter_names()))
    parser.add_argument('--pretrained-bert', dest='pretrained_bert', type=str, default=None)
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=10, nargs='?')
    parser.add_argument('--tokens-per-context', dest='tokens_per_context', type=int, default=128, nargs='?')
    parser.add_argument("--bert-framework", dest="bert_framework", type=str, default=None, choices=[None, "opennre"])
    parser.add_argument("--bert-torch-checkpoint", dest="bert_torch_checkpoint", type=str)
    parser.add_argument("--torch-num-workers", dest="torch_num_workers", type=int, default=0)
    parser.add_argument("--labels-fmt", dest="labels_fmt", default="u:0,p:1,n:2", type=str)
    parser.add_argument("--device-type", dest="device_type", type=str, default="cpu", help="Device type applicable for launching machine learning models")
    parser.add_argument("--backend", dest="backend", type=str, default=None, choices=[None, "d3js_graphs"])
    parser.add_argument("--label-names", dest="d3js_label_names", type=str, default="p:pos,n:neg,u:neu")
    parser.add_argument("--host", dest="d3js_host", default=None, type=str)
    parser.add_argument('--log-file', dest="log_file", default=None, type=str)
    parser.add_argument('-o', dest='output_template', type=str, default="output", nargs='?')

    return parser


def to_list(terms):
    r = []
    for t in terms:
        if isinstance(t, str):
            for i in t.split(' '):
                r.append(i)
        else:
            r.append(t)
    return r


if __name__ == '__main__':

    # Completing list of arguments.
    parser = create_infer_parser()

    # Parsing arguments.
    args = parser.parse_args()

    # Setup logger
    logger = setup_custom_logger(name="arelight", filepath=args.log_file)
    tqdm_log_out = TqdmToLogger(logger) if args.log_file is not None else None

    # Reading text-related parameters.
    sentence_parser = create_sentence_parser(framework=args.sentence_parser.split(":")[0],
                                             language=args.sentence_parser.split(":")[1])
    ner_framework = args.ner_framework
    ner_model_name = args.ner_model_name
    ner_object_types = args.ner_types
    terms_per_context = args.terms_per_context
    docs_limit = args.docs_limit
    output_template = args.output_template
    output_dir = dirname(args.output_template) if dirname(args.output_template) != "" else args.output_template

    assert(is_port_number(number=args.d3js_host, is_optional=True))

    # Classification task label scaler setup.
    labels_scl = {a: int(v) for a, v in map(lambda itm: itm.split(":"), args.labels_fmt.split(','))}
    labels_scaler = CustomLabelScaler(**labels_scl)

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

    collection_name = setup_collection_name(args.collection_name)

    collection_target_func = lambda data_type: join(output_dir, "-".join([collection_name, data_type.name.lower()]))

    sampling_engines_setup = {
        None: {},
        "arekit": {
            "rows_provider": create_bert_sample_provider(
                provider_type=SampleFormattersService.name_to_type(args.text_b_type) if args.text_b_type is not None else None,
                # We annotate everything with NoLabel.
                label_scaler=SingleLabelScaler(NoLabel()),
                entity_formatter=SharpPrefixedEntitiesSimpleFormatter(),
                crop_window=terms_per_context),
            "samples_io": CustomSamplesIO(create_target_func=collection_target_func,
                                          reader=SQliteReader(table_name="contents"),
                                          writer=SQliteWriter(table_name="contents")),
            "storage": RowCacheStorage(
                force_collect_columns=[const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND],
                log_out=tqdm_log_out),
            "save_labels_func": lambda data_type: data_type != DataType.Test
        }
    }
    
    translate_model = {
        None: lambda: None,
        "googletrans": lambda: create_translate_model("googletrans")
    }

    translator = translate_model[args.translate_framework]()

    stemmer_types = {
        None: lambda: None,
        "mystem": lambda: MystemWrapper()
    }

    stemmer = stemmer_types[args.stemmer]()

    def __entity_display_value(entity_value):
        """ This function describes how the result entity is expected to be visualized
            and passed for further components of the pipeline after serialization.
        """

        display_value = entity_value

        if stemmer is not None:
            display_value = stemmer.lemmatize_to_str(display_value)

        if args.translate_entity is not None:
            src, dest = args.translate_entity.split(':')
            display_value = translator([display_value], src=src, dest=dest)[0]

        return display_value

    entity_parsers = {
        # Default parser.
        None: lambda: TextEntitiesParser(id_assigner=IdAssigner(),
                                         display_value_func=__entity_display_value),
        # Parser based on DeepPavlov backend.
        "deeppavlov": lambda: DeepPavlovNERPipelineItem(
            src_func=lambda text: split_by_whitespaces(text),
            obj_filter=None if ner_object_types is None else lambda s_obj: s_obj.ObjectType in ner_object_types,
            ner_model_name=ner_model_name,
            id_assigner=IdAssigner(),
            display_value_func=__entity_display_value),
        # Parser based on Transformers backend.
        "transformers": lambda: TransformersNERPipelineItem(
            obj_filter=None if ner_object_types is None else lambda entity_group: entity_group in ner_object_types,
            ner_model_name=ner_model_name,
            id_assigner=IdAssigner(),
            display_value_func=__entity_display_value,
            device=args.device_type),
    }

    infer_engines_setup = {
        None: {},
        "opennre": {
            "data_loader_num_workers": args.torch_num_workers,
            "pretrained_bert": args.pretrained_bert,
            "checkpoint_path": args.bert_torch_checkpoint,
            "device_type": args.device_type,
            "max_seq_length": args.tokens_per_context,
            "batch_size": args.batch_size,
            "pooler": "cls",
            "predefined_ckpts": OPENNRE_CHECKPOINTS,
            "table_name": "contents",
            "logger": logger,
            "task_kwargs": {
                "no_label": str(labels_scaler.label_to_int(NoLabel())),
                "default_id_column": ID,
                "index_columns": [S_IND, T_IND],
                "text_columns": [PairTextProvider.TEXT_A, PairTextProvider.TEXT_B]
            },
        },
    }

    backend_setups = {
        None: {},
        "d3js_graphs": {
            "graph_min_links": 1,
            "graph_a_labels": None,
            "weights": True,
        }
    }

    predict_writers = {
        "tsv": TsvPredictWriter(log_out=tqdm_log_out),
        "sqlite3": SQLite3PredictWriter(table_name="open_nre_bert", log_out=tqdm_log_out)
    }

    predict_readers = {
        "tsv": PandasCsvReader(compression='infer'),
        "sqlite3": SQliteReader(table_name="open_nre_bert")
    }

    predict_extension = {
        "tsv": ".tsv.gz",
        "sqlite3": ".sqlite"
    }

    # Setup main pipeline.
    pipeline = demo_infer_texts_bert_pipeline(
        sampling_engines={key: sampling_engines_setup[key] for key in [args.sampling_framework]},
        infer_engines={key: infer_engines_setup[key] for key in [args.bert_framework]},
        backend_engines={key: backend_setups[key] for key in [args.backend]},
        inference_writer=predict_writers[args.inference_writer])

    # Settings.
    settings = []

    if args.sampling_framework == "arekit":

        synonyms_setup = {
            None: lambda: SimpleSynonymCollection(
                iter_group_values_lists=iter_group_values(args.synonyms_filepath),
                is_read_only=False),
            "lemmatized": lambda: StemmerBasedSynonymCollection(
                iter_group_values_lists=iter_group_values(args.synonyms_filepath),
                stemmer=stemmer,
                is_read_only=False)
        }

        text_translator_setup = {
            None: lambda: [],
            "ml-based": lambda: [
                MLTextTranslatorPipelineItem(
                    batch_translate_model=lambda content: translator(
                        str_list=content,
                        src=args.translate_text.split(':')[0],
                        dest=args.translate_text.split(':')[1]),
                    do_translate_entity=False),
                BasePipelineItem(src_func=lambda l: to_list(l)),
            ]
        }

        # Create Synonyms Collection.
        synonyms = synonyms_setup["lemmatized" if args.stemmer is not None else None]()

        # Text translator.
        tt = text_translator_setup["ml-based" if args.translate_text is not None else None]()
        print(type(tt))

        # Setup text parser.
        text_parser_pipeline = [
            BasePipelineItem(src_func=lambda s: s.Text),
            entity_parsers[ner_framework]()]\
        + tt + [
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ]

        # Reading from the optionally large list of files.
        doc_provider = CachedFilesDocProvider(
            filepaths=args.from_files,
            content_provider=lambda filepath: iter_content(
                filepath=filepath, csv_delimiter=args.csv_sep, csv_column=args.csv_column),
            content_to_sentences=sentence_parser,
            docs_limit=docs_limit)

        data_pipeline = create_neutral_annotation_pipeline(
            synonyms=synonyms,
            dist_in_terms_bound=terms_per_context,
            doc_provider=doc_provider,
            terms_per_context=terms_per_context,
            text_pipeline=text_parser_pipeline,
            batch_size=args.batch_size)

        settings.append({
            "data_type_pipelines": {DataType.Test: data_pipeline},
            "doc_ids": list(doc_provider.iter_doc_ids())
        })

    if args.backend == "d3js_graphs":
        labels_fmt = {a: v for a, v in map(lambda item: item.split(":"), args.d3js_label_names.split(','))}
        settings.append({
            "labels_formatter": CustomLabelsFormatter(**labels_fmt),
            "d3js_collection_name": collection_name,
            "d3js_collection_description": collection_name,
            "d3js_graph_output_dir": output_dir,
            "d3js_host": args.d3js_host,
        })

    settings.append({
        "labels_scaler": labels_scaler,
        # We provide this settings for inference.
        "predict_filepath": collection_target_func(DataType.Test) + predict_extension[args.inference_writer],
        "samples_io": sampling_engines_setup["arekit"]["samples_io"],
        "predict_reader": predict_readers[args.inference_writer]
    })

    # Launch application.
    BasePipelineLauncher.run(pipeline=pipeline, pipeline_ctx=PipelineResult(merge_dictionaries(settings)),
                             has_input=False)

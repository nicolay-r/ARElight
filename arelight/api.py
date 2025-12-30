from arekit.common.data import const
from arekit.common.data.const import ID
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection
from bulk_ner.src.pipeline.item.ner import BasePipelineItem, NERPipelineItem
from bulk_ner.src.utils import IdAssigner, setup_custom_logger
from bulk_translate.src.pipeline.translator import MLTextTranslatorPipelineItem
from arelight.arekit.indexed_entity import IndexedEntity
from arelight.arekit.samples_io import CustomSamplesIO
from arelight.arekit.utils_translator import string_terms_to_list
from arelight.const import BULK_CHAIN, D3JS_GRAPHS
from arelight.data.writers.sqlite_native import SQliteWriter
from arelight.doc_provider import CachedFilesDocProvider
from arelight.entity import HighligtedEntitiesFormatter
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_llm import demo_infer_texts_llm_pipeline
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.predict.writer_sqlite3 import SQLite3PredictWriter
from arelight.readers.csv_pd import PandasCsvReader
from arelight.readers.sqlite import SQliteReader
from arelight.run.utils import create_sentence_parser, iter_content, iter_group_values
from arelight.run.utils_logger import TqdmToLogger
from arelight.samplers.cropped import create_prompted_sample_provider
# TODO. These dependencies should be passed as parameters.
from arelight.third_party.dp_130 import DeepPavlovNER
from arelight.third_party.gt_310a import GoogleTranslateModel
from arelight.utils import flatten


def create_inference_pipeline(args, collection_target_func, predict_table_name):

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

    predict_extension = {
        "tsv": ".tsv.gz",
        "sqlite3": ".sqlite"
    }

    predict_readers = {
        "tsv": PandasCsvReader(compression='infer'),
        "sqlite3": SQliteReader(table_name=predict_table_name)
    }

    predict_writers = {
        "tsv": TsvPredictWriter(log_out=tqdm_log_out),
        "sqlite3": SQLite3PredictWriter(table_name=predict_table_name, log_out=tqdm_log_out)
    }

    sampling_engines_setup = {
        None: {},
        "arekit": {
            "rows_provider": create_prompted_sample_provider(
                label_scaler=SingleLabelScaler(NoLabel()),
                entity_formatter=HighligtedEntitiesFormatter(),
                is_entity_func=lambda term: isinstance(term, IndexedEntity),
                entity_group_ind_func=lambda entity: entity.GroupIndex,
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
        "googletrans": lambda: GoogleTranslateModel()
    }

    translator = translate_model[args.translate_framework]()

    # TODO. NERPipelineItem is common, while model is the only parameter that could be changed.
    entity_parsers = {
        # Parser based on DeepPavlov backend.
        "deeppavlov": lambda: NERPipelineItem(
            id_assigner=IdAssigner(),
            src_func=lambda text: split_by_whitespaces(text),
            model=DeepPavlovNER(model=ner_model_name, download=False, install=False),
            obj_filter=None if ner_object_types is None else lambda s_obj: s_obj.ObjectType in ner_object_types,
            # It is important to provide the correct type (see AREkit #575)
            create_entity_func=lambda value, e_type, entity_id: IndexedEntity(value=value, e_type=e_type, entity_id=entity_id),
            chunk_limit=128)
    }

    def class_to_int(text):
        if 'positive' in text.lower():
            return 1
        elif 'negative' in text.lower():
            return -1 
        return 0 


    infer_engines_setup = {
        None: {},
        BULK_CHAIN: {
            "class_name": args.inference_filename,
            "model_name": args.inference_model_name,
            "api_key": args.inference_api,
            "table_name": "contents",
            "task_kwargs": {
                "default_id_column": ID,
                "batch_size": args.batch_size,
                # TODO: concept for output structuring func
                "class_to_int": lambda row: class_to_int(row['response']),
                "prompt_schema": [{
                    "prompt": f"Given text: {{{BaseSingleTextProvider.TEXT_A}}}" +
                               f"TASK: Classify sentiment attitude of [SUBJECT] to [OBJECT]: "
                               f"positive, "
                               f"negative, "
                               f"neutral",
                     "out": "response"
                }]
            },
        },
    }

    backend_setups = {
        None: {},
        D3JS_GRAPHS: {
            "graph_min_links": 1,
            "graph_a_labels": None,
            "weights": True,
        }
    }

    # Setup main pipeline.
    pipeline = demo_infer_texts_llm_pipeline(
        sampling_engines={key: sampling_engines_setup[key] for key in [args.sampling_framework]},
        infer_engines={key: infer_engines_setup[key] for key in [args.inference_framework]},
        backend_engines={key: backend_setups[key] for key in [args.backend]},
        inference_writer=predict_writers[args.inference_writer])

    # Settings.
    settings = []

    if args.sampling_framework == "arekit":

        synonyms_setup = {
            None: lambda: SimpleSynonymCollection(
                iter_group_values_lists=iter_group_values(args.synonyms_filepath),
                is_read_only=False)
        }

        text_translator_setup = {
            None: lambda: None,
            "ml-based": lambda: [
                MLTextTranslatorPipelineItem(
                    batch_translate_model=translator.get_func(
                        src=args.translate_text.split(':')[0],
                        dest=args.translate_text.split(':')[1]),
                    do_translate_entity=False,
                    is_span_func=lambda term: isinstance(term, IndexedEntity)),
                BasePipelineItem(src_func=lambda l: string_terms_to_list(l)),
            ]
        }

        # Create Synonyms Collection.
        synonyms = synonyms_setup[None]()

        # Setup text parser.
        text_parser_pipeline = flatten([
            BasePipelineItem(src_func=lambda s: s.Text),
            entity_parsers[ner_framework](),
            text_translator_setup["ml-based" if args.translate_text is not None else None](),
            EntitiesGroupingPipelineItem(
                is_entity_func=lambda term: isinstance(term, IndexedEntity),
                value_to_group_id_func=lambda value:
                    SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                        synonyms=synonyms, value=value))
        ])

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

    # Classification task label scaler setup.
    labels_scl = {a: int(v) for a, v in map(lambda itm: itm.split(":"), args.labels_fmt.split(','))}
    labels_scaler = CustomLabelScaler(**labels_scl)

    settings.append({
        "labels_scaler": labels_scaler,
        # We provide these settings for inference.
        "predict_filepath": collection_target_func(DataType.Test) + predict_extension[args.inference_writer],
        "samples_io": sampling_engines_setup["arekit"]["samples_io"],
        "predict_reader": predict_readers[args.inference_writer]
    })

    return pipeline, settings
from arekit.common.data import const
from arekit.common.data.const import ID
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.utils import split_by_whitespaces
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection
from bulk_ner.src.pipeline.item.ner import BasePipelineItem, IdAssigner, NERPipelineItem
from bulk_translate.src.pipeline.translator import MLTextTranslatorPipelineItem

from arelight.arekit.api.data_type import DataType
from arelight.arekit.indexed_entity import IndexedEntity
from arelight.arekit.samples_io import CustomSamplesIO
from arelight.arekit.utils_translator import string_terms_to_list
from arelight.const import BULK_CHAIN
from arelight.data.writers.sqlite_native import SQliteWriter
from arelight.doc_provider import InMemoryRowsDocProvider
from arelight.entity import HighligtedEntitiesFormatter
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.pipelines.factory import build_pipeline
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.predict.writer_sqlite3 import SQLite3PredictWriter
from arelight.readers.csv_pd import PandasCsvReader
from arelight.readers.sqlite import SQliteReader
from arelight.run.utils import iter_group_values
from arelight.samplers.cropped import create_prompted_sample_provider
from arelight.utils import flatten, get_event_loop


def __setup_text_parser_pipeline(text_translator_func, entity_parser_func, synonyms):
    # Setup text parser.
    return flatten([
        BasePipelineItem(src_func=lambda s: s.Text),
        entity_parser_func(),
        text_translator_func(),
        EntitiesGroupingPipelineItem(
            is_entity_func=lambda term: isinstance(term, IndexedEntity),
            value_to_group_id_func=lambda value:
                SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
    ])


def _rows_as_dict(rows_iter):
    """ Materialize row dicts into the mapping `InMemoryRowsDocProvider` stores.

    A dict is already that mapping. Every other row must include `id`.
    """
    if isinstance(rows_iter, dict):
        return rows_iter
    rows = {}
    for value in rows_iter or []:
        rows[str(value["id"])] = value
    return rows


def _whole_text(text):
    return [text]


def __translator_pipeline(translator_args, event_loop):
    args = dict(translator_args)
    model = args.pop("model")
    do_translate_entity = args.pop("do_translate_entity", False)
    return [
        MLTextTranslatorPipelineItem(
            batch_translate_model=model.get_func(
                **(args | {"event_loop": event_loop})
            ),
            do_translate_entity=do_translate_entity,
            is_span_func=lambda term: isinstance(term, IndexedEntity)),
        BasePipelineItem(src_func=lambda l: string_terms_to_list(l)),
    ]


def create_inference_pipeline(rows_iter, sampling_args, predict_table_name,
                              collection_target_func, translator_args,
                              ner_args, inference_args,
                              tqdm_log_out=None, batch_size=1, event_loop=None,
                              sentence_parser=None):


    event_loop = get_event_loop() if event_loop is None else event_loop

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
                crop_window=sampling_args.get("terms_per_context", None)),
            "samples_io": CustomSamplesIO(create_target_func=collection_target_func,
                                          reader=SQliteReader(table_name="contents"),
                                          writer=SQliteWriter(table_name="contents")),
            "storage": RowCacheStorage(
                force_collect_columns=[const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND],
                log_out=tqdm_log_out),
            "save_labels_func": lambda data_type: data_type != DataType.Test
        }
    }

    entity_parsers = {
        # Parser based on DeepPavlov backend.
        "bulk-ner": lambda: NERPipelineItem(
            id_assigner=IdAssigner(),
            # It is important to provide the correct type (see AREkit #575)
            create_entity_func=lambda value, e_type, entity_id: IndexedEntity(value=value, e_type=e_type, entity_id=entity_id),
            # Pre-processing function for the text.
            src_func=lambda text: ner_args.get("src_func", split_by_whitespaces)(text),
            **ner_args)
    }

    inference_writer = inference_args.get("writer", "sqlite3")
    chain_args = {key: value for key, value in inference_args.items() if key != "writer"}

    infer_engines_setup = {
        None: {},
        BULK_CHAIN: {
            "table_name": "contents",
            **chain_args,
            "task_kwargs": {
                "default_id_column": ID,
                "batch_size": batch_size,
                "event_loop": event_loop,
            } | chain_args.get("task_kwargs", {}),
        },
    }

    pipeline = build_pipeline(
        sampling_engines={key: sampling_engines_setup[key] for key in ["arekit"]},
        infer_engines={key: infer_engines_setup[key] for key in [BULK_CHAIN]},
        inference_writer=predict_writers[inference_writer])

    # Settings.
    settings = []

    text_translator_setup = {
        None: lambda: [],
        "ml-based": lambda: __translator_pipeline(translator_args, event_loop),
    }

    synonyms_setup = {
        None: lambda: SimpleSynonymCollection(
            iter_group_values_lists=iter_group_values(sampling_args.get("synonyms_filepath", None)),
            is_read_only=False)
    }

    if sampling_args is not None:

        synonyms = synonyms_setup[None]()

        text_parser_pipeline = __setup_text_parser_pipeline(
            synonyms=synonyms,
            text_translator_func=text_translator_setup["ml-based" if translator_args else None],
            entity_parser_func=entity_parsers["bulk-ner"])

        # Sentence splitting is optional and supplied by the caller.
        # TODO. Sentence parser: concept of text structuring? / sentence / paragraphs etc.
        # TODO. We can disable that feature by default / or consider chunking_args (optional).
        content_to_sentences = _whole_text if sentence_parser is None else sentence_parser

        # The in-memory provider keeps one dict of document id to text.
        doc_provider = InMemoryRowsDocProvider(
            rows=_rows_as_dict(rows_iter),
            content_to_sentences=content_to_sentences,
            text_key=sampling_args.get("text_key", "text"),
            docs_limit=sampling_args.get("docs_limit", None))

        data_pipeline = create_neutral_annotation_pipeline(
            synonyms=synonyms,
            dist_in_terms_bound=sampling_args.get("dist_in_terms_bound", None),
            doc_provider=doc_provider,
            terms_per_context=sampling_args.get("terms_per_context", None),
            text_pipeline=text_parser_pipeline,
            batch_size=batch_size)

        settings.append({
            "data_type_pipelines": {DataType.Test: data_pipeline},
            "doc_ids": list(doc_provider.iter_doc_ids())
        })

    labels_fmt = inference_args['task_kwargs'].get("labels_fmt", None)

    # Classification task label scaler setup.
    labels_scl = {a: int(v) for a, v in map(lambda itm: itm.split(":"), labels_fmt.split(','))}

    settings.append({
        "labels_scaler": CustomLabelScaler(**labels_scl),
        # We provide these settings for inference.
        "predict_filepath": collection_target_func(DataType.Test) + predict_extension[inference_writer],
        "samples_io": sampling_engines_setup["arekit"]["samples_io"],
        "predict_reader": predict_readers[inference_writer]
    })

    return pipeline, settings
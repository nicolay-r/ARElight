from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.utils import split_by_whitespaces

import utils
from os.path import join, realpath, dirname

import unittest

from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.contrib.utils.data.writers.sqlite_native import SQliteWriter
from arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.common.data import const
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipelineLauncher
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection

from arelight.arekit.samples_io import CustomSamplesIO
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.demo.labels.scalers import CustomLabelScaler
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.predict.writer_csv import TsvPredictWriter
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes
from arelight.synonyms import iter_synonym_groups
from arelight.utils import IdAssigner, get_default_download_dir


class TestInfer(unittest.TestCase):

    # Declare input texts.
    texts = [
        # Text 1.
        """24 марта президент США Джо-Байден провел переговоры с
           лидерами стран Евросоюза в Брюсселе , вызвав внимание рынка и предположения о
           том, что Америке удалось уговорить ЕС совместно бойкотировать российские нефть
           и газ.  Европейский-Союз крайне зависим от России в плане поставок нефти и
           газа."""
    ]

    current_dir = dirname(realpath(__file__))
    TEST_DATA_DIR = join(current_dir, "data")

    @staticmethod
    def iter_groups(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for data in iter_synonym_groups(file):
                yield data

    def create_sampling_params(self):

        target_func = lambda data_type: join(utils.TEST_OUT_DIR, "-".join(["samples", data_type.name.lower()]))

        return {
            "rows_provider": create_bert_sample_provider(
                label_scaler=SingleLabelScaler(NoLabel()),
                provider_type=BertSampleProviderTypes.NLI_M,
                entity_formatter=SharpPrefixedEntitiesSimpleFormatter(),
                crop_window=50),
            "save_labels_func": lambda _: False,
            "samples_io": CustomSamplesIO(create_target_func=target_func, reader=JsonlReader(), writer=SQliteWriter()),
            "storage": RowCacheStorage(force_collect_columns=[
                const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND
            ])
        }

    def launch(self, pipeline):

        # We consider a texts[0] from the constant list.
        actual_content = self.texts

        synonyms = SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

        id_assigner = IdAssigner()

        # Setup text parsing.
        text_parser = [
            BasePipelineItem(src_func=lambda s: s.Text),
            DeepPavlovNERPipelineItem(ner_model_name="ner_ontonotes_bert_mult",
                                      src_func=lambda text: split_by_whitespaces(text),
                                      id_assigner=id_assigner,
                                      obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                                      chunk_limit=128),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ]

        data_pipeline = create_neutral_annotation_pipeline(
            synonyms=synonyms,
            dist_in_terms_bound=100,
            dist_in_sentences=0,
            doc_provider=utils.InMemoryDocProvider(docs=utils.input_to_docs(actual_content)),
            terms_per_context=50,
            text_pipeline=text_parser,
            batch_size=10)

        BasePipelineLauncher.run(pipeline=pipeline,
                                 pipeline_ctx=PipelineContext(d={
                                     "labels_scaler": CustomLabelScaler(),
                                     "predict_filepath": join(utils.TEST_OUT_DIR, "predict.tsv.gz"),
                                     "data_type_pipelines": {DataType.Test: data_pipeline},
                                     "doc_ids": list(range(len(actual_content))),
                                 }),
                                 has_input=False)

    def test_opennre(self):

        pipeline = demo_infer_texts_bert_pipeline(
            inference_writer=TsvPredictWriter(),
            sampling_engines={
                "arekit": self.create_sampling_params()
            },
            infer_engines={
                "opennre": {
                    "pretrained_bert": "DeepPavlov/rubert-base-cased",
                    "checkpoint_path": join(get_default_download_dir(), "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar"),
                    "device_type": "cpu",
                    "max_seq_length": 128,
                    "task_kwargs": {
                        "no_label": "0",
                        "default_id_column": "id",
                        "index_columns": ["s_ind", "t_ind"],
                        "text_columns": ["text_a", "text_b"]
                    },
        }
        })

        self.launch(pipeline)

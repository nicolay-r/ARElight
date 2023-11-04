from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.utils.data.readers.jsonl import JsonlReader
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter

import utils
from os.path import join, realpath, dirname

import unittest

from arekit.common.data import const
from arekit.common.pipeline.context import PipelineContext
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.common.docs.base import Document
from arekit.common.docs.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.docs.sentence import BaseDocumentSentence
from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.base import BasePipeline
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.source.synonyms.utils import iter_synonym_groups
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.synonyms.simple import SimpleSynonymCollection
from ru_sent_tokenize import ru_sent_tokenize

from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.items.entities_ner_dp import DeepPavlovNERPipelineItem
from arelight.run.utils import create_labels_scaler
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import BertSampleProviderTypes
from arelight.utils import IdAssigner


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

    @staticmethod
    def input_to_docs(texts):
        assert(isinstance(texts, list))
        docs = []
        for doc_id, contents in enumerate(texts):
            sentences = ru_sent_tokenize(contents)
            sentences = list(map(lambda text: BaseDocumentSentence(text), sentences))
            doc = Document(doc_id=doc_id, sentences=sentences)
            docs.append(doc)
        return docs

    def create_sampling_params(self):
        writer = OpenNREJsonWriter(
            text_columns=[BaseSingleTextProvider.TEXT_A, PairTextProvider.TEXT_B],
            keep_extra_columns=False,
            na_value="0")

        return {
            "rows_provider": create_bert_sample_provider(
                label_scaler=SingleLabelScaler(NoLabel()),
                provider_type=BertSampleProviderTypes.NLI_M,
                entity_formatter=SharpPrefixedEntitiesSimpleFormatter()),
            "save_labels_func": lambda _: False,
            "samples_io": SamplesIO(target_dir=utils.TEST_OUT_DIR,
                                    reader=JsonlReader(),
                                    writer=writer),
            "storage": RowCacheStorage(force_collect_columns=[
                const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND
            ])
        }

    def launch(self, pipeline):

        # We consider a texts[0] from the constant list.
        actual_content = self.texts

        pipeline = BasePipeline(pipeline)
        synonyms = SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

        id_assigner = IdAssigner()

        # Setup text parsing.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            DeepPavlovNERPipelineItem(ner_model_name="ner_ontonotes_bert_mult",
                                      id_assigner=id_assigner,
                                      obj_filter=lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
                                      chunk_limit=128),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ])

        data_pipeline = create_neutral_annotation_pipeline(
            synonyms=synonyms,
            dist_in_terms_bound=100,
            dist_in_sentences=0,
            doc_provider=utils.InMemoryDocProvider(docs=self.input_to_docs(actual_content)),
            terms_per_context=50,
            text_parser=text_parser)

        pipeline.run(input_data=PipelineContext(d={
            "labels_scaler": create_labels_scaler(3),
            "predict_filepath": join(utils.TEST_OUT_DIR, "predict.tsv.gz"),
            "data_type_pipelines": {DataType.Test: data_pipeline},
            "doc_ids": list(range(len(actual_content))),
        }))

    def test_opennre(self):

        pipeline = demo_infer_texts_bert_pipeline(
            sampling_engines={
                "arekit": self.create_sampling_params()
            },
            infer_engines={
                "opennre": {
                    "pretrained_bert": "DeepPavlov/rubert-base-cased",
                    "checkpoint_path": "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar",
                    "device_type": "cpu",
                    "max_seq_length": 128
                }
            })

        self.launch(pipeline)

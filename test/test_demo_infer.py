import unittest
from os.path import join

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

from arelight.doc_provider import InMemoryDocProvider
from arelight.pipelines.data.annot_pairs_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert import demo_infer_texts_bert_pipeline
from arelight.pipelines.items.id_assigner import IdAssigner
from arelight.run.args import const
from arelight.run.args.common import create_entity_parser
from arelight.run.entities.factory import create_entity_formatter
from arelight.run.entities.types import EntityFormatterTypes
from arelight.run.utils import create_labels_scaler


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

    def launch(self, pipeline):

        # We consider a texts[0] from the constant list.
        actual_content = self.texts

        pipeline = BasePipeline(pipeline)
        synonyms = SimpleSynonymCollection(iter_group_values_lists=[], is_read_only=False)

        id_assigner = IdAssigner()
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            create_entity_parser(ner_model_name="ner_ontonotes_bert_mult",
                                 id_assigner=id_assigner,
                                 obj_filter_types=["ORG", "PERSON", "LOC", "GPE"]),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ])

        data_pipeline = create_neutral_annotation_pipeline(
            synonyms=synonyms,
            dist_in_terms_bound=100,
            dist_in_sentences=0,
            doc_ops=InMemoryDocProvider(docs=self.input_to_docs(actual_content)),
            terms_per_context=50,
            text_parser=text_parser)

        pipeline.run(None, {
            "template_filepath": join(const.DATA_DIR, "brat_template.html"),
            "data_type_pipelines": {DataType.Test: data_pipeline},
            "doc_ids": list(range(len(actual_content))),
        })

    def test_deeppavlov(self):

        pipeline = demo_infer_texts_bert_pipeline(
            samples_output_dir="./data",
            samples_prefix="samples",
            pretrained_bert="bert-base-uncased",
            bert_type="deeppavlov",
            entity_fmt=create_entity_formatter(EntityFormatterTypes.HiddenBertStyled),
            labels_scaler=create_labels_scaler(3),
            bert_config_path=None,
            max_seq_length=None,
            checkpoint_path=None)

        self.launch(pipeline)

    def test_opennre(self):

        pipeline = demo_infer_texts_bert_pipeline(
            samples_output_dir="./data",
            samples_prefix="samples",
            pretrained_bert="DeepPavlov/rubert-base-cased",
            entity_fmt=create_entity_formatter(EntityFormatterTypes.HiddenBertStyled),
            labels_scaler=create_labels_scaler(3),
            bert_config_path=None,
            max_seq_length=128,
            bert_type="opennre",
            checkpoint_path="../data/ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar")

        self.launch(pipeline)

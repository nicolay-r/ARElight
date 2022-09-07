import json
import unittest
from os.path import dirname, realpath, join

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.enum_input_types import ModelInputType
from arekit.contrib.networks.enum_name_types import ModelNames
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter, \
    RuSentiFramesEffectLabelsFormatter
from arekit.contrib.source.rusentiframes.types import RuSentiFramesVersions
from arekit.contrib.utils.entities.formatters.str_simple_fmt import StringEntitiesSimpleFormatter
from arekit.contrib.utils.entities.formatters.str_simple_sharp_prefixed_fmt import SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.utils.pipelines.items.text.frames import FrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper

from arelight.doc_ops import InMemoryDocOperations
from arelight.network.nn.common import create_and_fill_variant_collection
from arelight.pipelines.annot_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.infer_bert_rus import demo_infer_texts_bert_pipeline
from arelight.pipelines.demo.infer_nn_rus import demo_infer_texts_tensorflow_nn_pipeline
from arelight.pipelines.demo.labels.base import NegativeLabel, PositiveLabel
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler
from arelight.pipelines.demo.utils import read_synonyms_collection
from arelight.pipelines.items.entities_bert_ontonotes import BertOntonotesNERPipelineItem
from arelight.pipelines.items.utils import input_to_docs

from examples.args import const


class TestDemo(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    TEST_DATA_DIR = join(current_dir, "data")
    ORIGIN_DATA_DIR = join(current_dir, "../data")

    text = "США вводит санкции против РФ"

    @staticmethod
    def __prepare_template(data, text, template_filepath, brat_url):
        assert(isinstance(brat_url, str))
        assert(isinstance(data, dict))

        with open(template_filepath, "r") as template_file:
            template_local = template_file.read()

        template_local = template_local.replace("$____COL_DATA_SEM____", json.dumps(data.get('coll_data', '')))
        template_local = template_local.replace("$____DOC_DATA_SEM____", json.dumps(data.get('doc_data', '')))
        template_local = template_local.replace("$____BRAT_URL____", brat_url)
        template_local = template_local.replace("$____TEXT____", text)

        return template_local

    def test_demo_rus_nn(self):
        frames_collection = RuSentiFramesCollection.read_collection(
            version=RuSentiFramesVersions.V20,
            labels_fmt=RuSentiFramesLabelsFormatter(
                pos_label_type=PositiveLabel, neg_label_type=NegativeLabel),
            effect_labels_fmt=RuSentiFramesEffectLabelsFormatter(
                pos_label_type=PositiveLabel, neg_label_type=NegativeLabel))

        demo_pipeline = demo_infer_texts_tensorflow_nn_pipeline(
            texts_count=1,
            output_dir=".",
            model_name=ModelNames.PCNN,
            entity_fmt=StringEntitiesSimpleFormatter(),
            model_load_dir=const.NEURAL_NETWORKS_TARGET_DIR,
            model_input_type=ModelInputType.SingleInstance,
            frames_collection=frames_collection)

        stemmer = MystemWrapper()

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

        synonyms = read_synonyms_collection(synonyms_filepath=join(self.ORIGIN_DATA_DIR, "synonyms.txt"),
                                            stemmer=stemmer)

        single_doc = [self.text.strip()]
        doc_ops = InMemoryDocOperations(docs=input_to_docs(single_doc))

        # Initialize data processing pipeline.
        data_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                           dist_in_terms_bound=50,
                                                           terms_per_context=50,
                                                           doc_ops=doc_ops,
                                                           text_parser=text_parser,
                                                           dist_in_sentences=0)

        contents = demo_pipeline.run(None, {
            "template_filepath": join(const.DATA_DIR, "brat_template.html"),
            "data_folding": NoFolding(doc_ids=[0], supported_data_type=DataType.Test),
            "data_type_pipelines": {DataType.Test: data_pipeline}
        })

        template = TestDemo.__prepare_template(
            data=contents, text=self.text,
            template_filepath=join(TestDemo.ORIGIN_DATA_DIR, "brat_template.html"),
            brat_url="http://localhost:8001/")

        with open(join(self.TEST_DATA_DIR, "demo-rus-nn-output.html"), "w") as output:
            output.write(template)

    def test_demo_rus_bert(self):

        model_dir = join(self.ORIGIN_DATA_DIR, "models")
        state_name = "ra-20-srubert-large-neut-nli-pretrained-3l"
        finetuned_state_name = "ra-20-srubert-large-neut-nli-pretrained-3l-finetuned"

        demo_pipeline = demo_infer_texts_bert_pipeline(
            texts_count=1,
            output_dir=".",
            labels_scaler=ThreeLabelScaler(),
            entity_fmt=SharpPrefixedEntitiesSimpleFormatter(),
            bert_vocab_path=join(model_dir, state_name, "vocab.txt"),
            bert_config_path=join(model_dir, state_name, "bert_config.json"),
            bert_finetuned_ckpt_path=join(model_dir, finetuned_state_name, state_name))

        single_doc = [self.text.strip()]
        doc_ops = InMemoryDocOperations(docs=input_to_docs(single_doc))

        synonyms = read_synonyms_collection(synonyms_filepath=join(self.ORIGIN_DATA_DIR, "synonyms.txt"),
                                            stemmer=MystemWrapper())

        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            BertOntonotesNERPipelineItem(lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"]),
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    synonyms=synonyms, value=value))
        ])

        data_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms,
                                                           terms_per_context=50,
                                                           dist_in_terms_bound=50,
                                                           doc_ops=doc_ops,
                                                           text_parser=text_parser)

        contents = demo_pipeline.run(None, {
            "template_filepath": join(const.DATA_DIR, "brat_template.html"),
            "data_folding": NoFolding(doc_ids=[0], supported_data_type=DataType.Test),
            "data_type_pipelines": {DataType.Test: data_pipeline}
        })

        template = TestDemo.__prepare_template(
            data=contents, text=self.text,
            template_filepath=join(TestDemo.ORIGIN_DATA_DIR, "brat_template.html"),
            brat_url="http://localhost:8001/")

        with open(join(self.TEST_DATA_DIR, "demo-rus-bert-output.html"), "w") as output:
            output.write(template)


if __name__ == '__main__':
    unittest.main()

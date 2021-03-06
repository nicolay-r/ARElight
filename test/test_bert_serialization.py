import unittest
from os.path import dirname, join, realpath

from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.base import News
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.news.sentence import BaseNewsSentence
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.handlers.serializer import BertExperimentInputSerializerIterationHandler
from arekit.contrib.bert.samplers.types import BertSampleProviderTypes
from arekit.contrib.experiment_rusentrel.entities.str_simple_sharp_prefixed_fmt import \
    SharpPrefixedEntitiesSimpleFormatter
from arekit.contrib.experiment_rusentrel.synonyms.collection import StemmerBasedSynonymCollection
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.text.pipeline_terms_splitter import TermsSplitterParser
from ru_sent_tokenize import ru_sent_tokenize

from arelight.exp.doc_ops import CustomDocOperations
from arelight.exp.exp_io import InferIOUtils
from arelight.exp.opin_ops import CustomOpinionOperations
from arelight.network.bert.ctx import BertSerializationContext
from arelight.pipelines.utils import input_to_docs
from arelight.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem


class BertTestSerialization(unittest.TestCase):
    """ This unit test represent a tutorial on how
        AREkit might be applied towards data preparation for BERT
        model.
    """

    current_dir = dirname(realpath(__file__))
    ORIGIN_DATA_DIR = join(current_dir, "../data")
    TEST_DATA_DIR = join(current_dir, "data")

    @staticmethod
    def get_synonym_group_index(s, value):
        if not s.contains_synonym_value(value):
            s.add_synonym_value(value)
        return s.get_synonym_group_index(value)

    @staticmethod
    def input_to_docs(texts):
        docs = []
        for doc_id, contents in enumerate(texts):
            sentences = ru_sent_tokenize(contents)
            sentences = list(map(lambda text: BaseNewsSentence(text), sentences))
            doc = News(doc_id=doc_id, sentences=sentences)
            docs.append(doc)
        return docs

    @staticmethod
    def iter_groups(filepath):
        with open(filepath, 'r', encoding='utf-8') as file:
            for data in iter_synonym_groups(file):
                yield data

    def test(self):

        # Declare input texts.
        texts = [
            # Text 1.
            """24 ?????????? ?????????????????? ?????? ?????? ???????????? ???????????? ???????????????????? ??
               ???????????????? ?????????? ?????????????????? ?? ????????????????, ???????????? ???????????????? ?????????? ?? ?????????????????????????? ??
               ??????, ?????? ?????????????? ?????????????? ?????????????????? ???? ?????????????????? ?????????????????????????? ???????????????????? ??????????
               ?? ??????.  ?????????????????????? ???????? ???????????? ?????????????? ???? ???????????? ?? ?????????? ???????????????? ?????????? ??
               ????????."""
        ]

        # Declare synonyms collection.
        synonyms_filepath = join(self.ORIGIN_DATA_DIR, "synonyms.txt")

        synonyms = StemmerBasedSynonymCollection(
            iter_group_values_lists=self.iter_groups(synonyms_filepath),
            stemmer=MystemWrapper(),
            is_read_only=False,
            debug=False)

        # Declare text parser.
        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            BertOntonotesNERPipelineItem(lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"]),
            EntitiesGroupingPipelineItem(lambda value: self.get_synonym_group_index(synonyms, value))
        ])

        # Declaring algo.
        algo = PairBasedAnnotationAlgorithm(
            label_provider=ConstantLabelProvider(label_instance=NoLabel()),
            dist_in_terms_bound=None)

        # Declare folding and experiment context.
        no_folding = NoFolding(doc_ids_to_fold=list(range(len(texts))),
                               supported_data_types=[DataType.Test])
        exp_ctx = BertSerializationContext(
            label_scaler=SingleLabelScaler(NoLabel()),
            annotator=DefaultAnnotator(algo),
            terms_per_context=50,
            str_entity_formatter=SharpPrefixedEntitiesSimpleFormatter(),
            name_provider=ExperimentNameProvider(name="example-bert", suffix="serialize"),
            data_folding=no_folding)

        # Composing labels formatter and experiment preparation.
        labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})
        exp_io = InferIOUtils(exp_ctx=exp_ctx, output_dir=self.TEST_DATA_DIR)
        doc_ops = CustomDocOperations(exp_ctx, text_parser=text_parser)
        opin_ops = CustomOpinionOperations(
            labels_formatter=labels_fmt,
            exp_io=exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=labels_fmt)

        exp = BaseExperiment(exp_io=exp_io, exp_ctx=exp_ctx, doc_ops=doc_ops, opin_ops=opin_ops)

        handler = BertExperimentInputSerializerIterationHandler(
            exp_io=exp_io,
            exp_ctx=exp_ctx,
            doc_ops=doc_ops,
            opin_ops=exp.OpinionOperations,
            ###
            sample_provider_type=BertSampleProviderTypes.NLI_M,     # TODO. Part of TEXT_B
            sample_labels_fmt=labels_fmt,                           # TODO. Part of TEXT_B
            ###
            annot_labels_fmt=labels_fmt,                            # TODO. To be removed in further.
            entity_formatter=exp_ctx.StringEntityFormatter,
            value_to_group_id_func=synonyms.get_synonym_group_index,
            balance_train_samples=True)

        # Initilize documents.
        docs = input_to_docs(texts)
        doc_ops.set_docs(docs)

        # Run.
        engine = ExperimentEngine(exp_ctx.DataFolding)  # Present folding limitation.
        engine.run([handler])


if __name__ == '__main__':
    unittest.main()

from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.bert.handlers.serializer import BertExperimentInputSerializerIterationHandler
from arekit.contrib.bert.samplers.types import BertSampleProviderTypes
from arekit.processing.text.pipeline_terms_splitter import TermsSplitterParser

from exp.doc_ops import CustomDocOperations
from exp.exp import CustomExperiment
from exp.exp_io import InferIOUtils
from network.bert.ctx import BertSerializationContext
from pipelines.utils import input_to_docs


class BertTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, terms_per_context, entities_parser, synonyms, opin_annot, name_provider,
                 entity_fmt, data_folding):
        assert(isinstance(entities_parser, BasePipelineItem))
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(name_provider, ExperimentNameProvider))

        # Label provider setup.
        labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})

        self.__exp_ctx = BertSerializationContext(
            label_scaler=SingleLabelScaler(NoLabel()),
            annotator=opin_annot,
            terms_per_context=terms_per_context,
            str_entity_formatter=entity_fmt,
            name_provider=name_provider,
            data_folding=data_folding)

        self.__exp_io = InferIOUtils(self.__exp_ctx)

        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            entities_parser,
            EntitiesGroupingPipelineItem(lambda value: self.get_synonym_group_index(synonyms, value))])

        self.__doc_ops = CustomDocOperations(exp_ctx=self.__exp_ctx,
                                             text_parser=text_parser)

        exp = CustomExperiment(
            synonyms=synonyms,
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            labels_formatter=labels_fmt,
            neutral_labels_fmt=labels_fmt)

        self.__handler = BertExperimentInputSerializerIterationHandler(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            opin_ops=exp.OpinionOperations,
            sample_labels_fmt=labels_fmt,
            annot_labels_fmt=labels_fmt,
            sample_provider_type=BertSampleProviderTypes.NLI_M,
            entity_formatter=self.__exp_ctx.StringEntityFormatter,
            value_to_group_id_func=synonyms.get_synonym_group_index,
            balance_train_samples=True)

    @staticmethod
    def get_synonym_group_index(synonyms, value):
        assert(isinstance(synonyms, SynonymsCollection))
        if not synonyms.contains_synonym_value(value):
            synonyms.add_synonym_value(value)
        return synonyms.get_synonym_group_index(value)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))

        docs = input_to_docs(input_data)

        # Setup document.
        self.__doc_ops.set_docs(docs)

        engine = ExperimentEngine(self.__exp_ctx.DataFolding)
        engine.run([self.__handler])

        return self.__exp_io

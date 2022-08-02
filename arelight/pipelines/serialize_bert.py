from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser

from arelight.exp.doc_ops import CustomDocOperations
from arelight.exp.exp_io import InferIOUtils
from arelight.exp.opin_ops import CustomOpinionOperations
from arelight.network.bert.ctx import BertSerializationContext
from arelight.pipelines.utils import input_to_docs


class BertTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, terms_per_context, entities_parser, synonyms, name_provider,
                 entity_fmt, text_b_type, data_folding, output_dir):
        assert(isinstance(entities_parser, BasePipelineItem))
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(text_b_type, BertSampleProviderTypes))
        assert(isinstance(name_provider, ExperimentNameProvider))
        assert(isinstance(output_dir, str))

        # Label provider setup.
        labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})

        self.__exp_ctx = BertSerializationContext(label_scaler=SingleLabelScaler(NoLabel()),
                                                  terms_per_context=terms_per_context,
                                                  name_provider=name_provider)

        self.__exp_io = InferIOUtils(exp_ctx=self.__exp_ctx, output_dir=output_dir)

        text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            entities_parser,
            EntitiesGroupingPipelineItem(lambda value: self.get_synonym_group_index(synonyms, value))])

        self.__doc_ops = CustomDocOperations(exp_ctx=self.__exp_ctx,
                                             text_parser=text_parser)

        self.__opin_ops = CustomOpinionOperations(
            labels_formatter=labels_fmt,
            exp_io=self.__exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=labels_fmt)

        exp = BaseExperiment(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            opin_ops=self.__opin_ops)

        # TODO. Handler does not exist anymore.
        self.__handler = BertExperimentInputSerializerIterationHandler(
            exp_io=self.__exp_io,
            exp_ctx=self.__exp_ctx,
            doc_ops=self.__doc_ops,
            opin_ops=exp.OpinionOperations,
            sample_labels_fmt=labels_fmt,
            annot_labels_fmt=labels_fmt,
            sample_provider_type=text_b_type,
            entity_formatter=entity_fmt,
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

        # TODO. Engine does not exist anymore.
        engine = ExperimentEngine(self.__exp_ctx.DataFolding)
        engine.run([self.__handler])

        return self.__exp_io

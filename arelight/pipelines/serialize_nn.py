from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms.base import SynonymsCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.contrib.utils.pipelines.items.text.frames import FrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015

from arelight.exp.doc_ops import CustomDocOperations
from arelight.exp.exp_io import InferIOUtils
from arelight.exp.opin_ops import CustomOpinionOperations
from arelight.network.nn.common import create_and_fill_variant_collection
from arelight.network.nn.ctx import NetworkSerializationContext
from arelight.pipelines.utils import input_to_docs


# TODO. This become a part of AREkit.
# TODO. This become a part of AREkit.
# TODO. This become a part of AREkit.
# TODO. This should be removed.
class NetworkTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, terms_per_context, entities_parser, synonyms, name_provider,
                 frames_collection, entity_fmt, stemmer, data_folding, output_dir, embedding=None):
        assert(isinstance(frames_collection, RuSentiFramesCollection))
        assert(isinstance(entities_parser, BasePipelineItem))
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(name_provider, ExperimentNameProvider))
        assert(isinstance(output_dir, str))

        # Initialize embedding.
        if embedding is None:
            self.__embedding = load_embedding_news_mystem_skipgram_1000_20_2015()

        # Initialize synonyms collection.
        self.__synonyms = synonyms
        pos_tagger = POSMystemWrapper(MystemWrapper().MystemInstance)

        # Label provider setup.
        self.__labels_fmt = StringLabelsFormatter(stol={"neu": NoLabel})

        # Initialize text parser with the related dependencies.
        frame_variants_collection = create_and_fill_variant_collection(frames_collection)
        self.__text_parser = BaseTextParser(pipeline=[
            TermsSplitterParser(),
            entities_parser,
            EntitiesGroupingPipelineItem(
                lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                    self.__synonyms, value)),
            DefaultTextTokenizer(keep_tokens=True),
            FrameVariantsParser(frame_variants=frame_variants_collection),
            LemmasBasedFrameVariantsParser(save_lemmas=False,
                                           stemmer=stemmer,
                                           frame_variants=frame_variants_collection),
            FrameVariantsSentimentNegation()])

        # initialize experiment related data.
        self.__exp_ctx = NetworkSerializationContext(labels_scaler=SingleLabelScaler(NoLabel()),
                                                     name_provider=name_provider)

        self.__exp_io = InferIOUtils(exp_ctx=self.__exp_ctx, output_dir=output_dir)

        self.__doc_ops = CustomDocOperations(exp_ctx=self.__exp_ctx,
                                             text_parser=self.__text_parser)

        self.__opin_ops = CustomOpinionOperations(
            labels_formatter=self.__labels_fmt,
            exp_io=self.__exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=self.__labels_fmt)

        # TODO. Remove this.
        self.__exp = BaseExperiment(exp_io=self.__exp_io,
                                    exp_ctx=self.__exp_ctx,
                                    doc_ops=self.__doc_ops,
                                    opin_ops=self.__opin_ops)

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, list))

        docs = input_to_docs(input_data)

        # Setup document.
        self.__doc_ops.set_docs(docs)

        # TODO. This is outdated.
        NetworkInputHelper.prepare(exp_ctx=self.__exp.ExperimentContext,
                                   exp_io=self.__exp.ExperimentIO,
                                   doc_ops=self.__exp.DocumentOperations,
                                   opin_ops=self.__exp.OpinionOperations,
                                   terms_per_context=self.__exp_ctx.TermsPerContext,
                                   balance=False,
                                   value_to_group_id_func=self.__synonyms.get_synonym_group_index)

        return self.__exp.ExperimentIO

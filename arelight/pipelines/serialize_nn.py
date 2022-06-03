from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.api.base import BaseExperiment
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.base import BaseDataFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.single import SingleLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.pipeline.items.base import BasePipelineItem
from arekit.common.synonyms import SynonymsCollection
from arekit.common.text.parser import BaseTextParser
from arekit.common.text.stemmer import Stemmer
from arekit.contrib.networks.core.input.helper import NetworkInputHelper
from arekit.contrib.source.rusentiframes.collection import RuSentiFramesCollection
from arekit.processing.lemmatization.mystem import MystemWrapper
from arekit.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.processing.text.pipeline_frames import FrameVariantsParser
from arekit.processing.text.pipeline_frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.processing.text.pipeline_frames_negation import FrameVariantsSentimentNegation
from arekit.processing.text.pipeline_terms_splitter import TermsSplitterParser
from arekit.processing.text.pipeline_tokenizer import DefaultTextTokenizer

from arelight.exp.doc_ops import CustomDocOperations
from arelight.exp.exp_io import InferIOUtils
from arelight.exp.opin_ops import CustomOpinionOperations
from arelight.network.nn.common import create_and_fill_variant_collection
from arelight.network.nn.ctx import NetworkSerializationContext
from arelight.network.nn.embedding import RusvectoresEmbedding
from arelight.pipelines.utils import input_to_docs


class NetworkTextsSerializationPipelineItem(BasePipelineItem):

    def __init__(self, terms_per_context, entities_parser, synonyms, opin_annot, name_provider,
                 embedding_path, frames_collection, entity_fmt, stemmer, data_folding, output_dir):
        assert(isinstance(frames_collection, RuSentiFramesCollection))
        assert(isinstance(entities_parser, BasePipelineItem))
        assert(isinstance(entity_fmt, StringEntitiesFormatter))
        assert(isinstance(synonyms, SynonymsCollection))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(embedding_path, str))
        assert(isinstance(stemmer, Stemmer))
        assert(isinstance(data_folding, BaseDataFolding))
        assert(isinstance(name_provider, ExperimentNameProvider))
        assert(isinstance(output_dir, str))

        # Initalize embedding.
        embedding = RusvectoresEmbedding.from_word2vec_format(filepath=embedding_path, binary=True)
        embedding.set_stemmer(stemmer)

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
            EntitiesGroupingPipelineItem(lambda value: self.get_synonym_group_index(self.__synonyms, value)),
            DefaultTextTokenizer(keep_tokens=True),
            FrameVariantsParser(frame_variants=frame_variants_collection),
            LemmasBasedFrameVariantsParser(save_lemmas=False,
                                           stemmer=stemmer,
                                           frame_variants=frame_variants_collection),
            FrameVariantsSentimentNegation()])

        # initialize expriment related data.
        self.__exp_ctx = NetworkSerializationContext(
            labels_scaler=SingleLabelScaler(NoLabel()),
            embedding=embedding,
            annotator=opin_annot,
            terms_per_context=terms_per_context,
            str_entity_formatter=entity_fmt,
            pos_tagger=pos_tagger,
            name_provider=name_provider,
            frames_collection=frames_collection,
            frame_variant_collection=frame_variants_collection,
            data_folding=data_folding)

        self.__exp_io = InferIOUtils(exp_ctx=self.__exp_ctx, output_dir=output_dir)

        self.__doc_ops = CustomDocOperations(exp_ctx=self.__exp_ctx,
                                             text_parser=self.__text_parser)

        self.__opin_ops = CustomOpinionOperations(
            labels_formatter=self.__labels_fmt,
            exp_io=self.__exp_io,
            synonyms=synonyms,
            neutral_labels_fmt=self.__labels_fmt)

        self.__exp = BaseExperiment(exp_io=self.__exp_io,
                                    exp_ctx=self.__exp_ctx,
                                    doc_ops=self.__doc_ops,
                                    opin_ops=self.__opin_ops)

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

        NetworkInputHelper.prepare(exp_ctx=self.__exp.ExperimentContext,
                                   exp_io=self.__exp.ExperimentIO,
                                   doc_ops=self.__exp.DocumentOperations,
                                   opin_ops=self.__exp.OpinionOperations,
                                   terms_per_context=self.__exp_ctx.TermsPerContext,
                                   balance=False,
                                   value_to_group_id_func=self.__synonyms.get_synonym_group_index)

        return self.__exp.ExperimentIO

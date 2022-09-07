import argparse
from os.path import join

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.news.entities_grouping import EntitiesGroupingPipelineItem
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.pipeline.base import BasePipeline
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.common.text.parser import BaseTextParser
from arekit.contrib.networks.core.input.ctx_serialization import NetworkSerializationContext
from arekit.contrib.networks.core.input.term_types import TermTypes
from arekit.contrib.utils.connotations.rusentiframes_sentiment import RuSentiFramesConnotationProvider
from arekit.contrib.utils.io_utils.embedding import NpEmbeddingIO
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.networks import NetworksInputSerializerPipelineItem
from arekit.contrib.utils.pipelines.items.text.frames import FrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_lemmatized import LemmasBasedFrameVariantsParser
from arekit.contrib.utils.pipelines.items.text.frames_negation import FrameVariantsSentimentNegation
from arekit.contrib.utils.pipelines.items.text.terms_splitter import TermsSplitterParser
from arekit.contrib.utils.pipelines.items.text.tokenizer import DefaultTextTokenizer
from arekit.contrib.utils.processing.lemmatization.mystem import MystemWrapper
from arekit.contrib.utils.processing.pos.mystem_wrap import POSMystemWrapper
from arekit.contrib.utils.resources import load_embedding_news_mystem_skipgram_1000_20_2015
from arekit.contrib.utils.vectorizers.bpe import BPEVectorizer
from arekit.contrib.utils.vectorizers.random_norm import RandomNormalVectorizer

from arelight.doc_ops import InMemoryDocOperations
from arelight.network.nn.common import create_and_fill_variant_collection
from arelight.pipelines.annot_nolabel import create_neutral_annotation_pipeline
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler
from arelight.pipelines.items.utils import input_to_docs

from examples.args import const
from examples.args import common
from examples.args.const import DEFAULT_TEXT_FILEPATH
from examples.entities.factory import create_entity_formatter
from examples.utils import read_synonyms_collection, create_labels_scaler

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    common.InputTextArg.add_argument(parser, default=None)
    common.FromFilesArg.add_argument(parser, default=[DEFAULT_TEXT_FILEPATH])
    common.SynonymsCollectionFilepathArg.add_argument(parser, default=join(const.DATA_DIR, "synonyms.txt"))
    common.EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    common.TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    common.EntityFormatterTypesArg.add_argument(parser, default="hidden-simple-eng")
    common.StemmerArg.add_argument(parser, default="mystem")
    common.FramesColectionArg.add_argument(parser)

    # Parsing arguments.
    args = parser.parse_args()

    text_from_arg = common.InputTextArg.read_argument(args)
    texts_from_files = common.FromFilesArg.read_argument(args)
    input_texts = text_from_arg if text_from_arg is not None else texts_from_files
    synonyms_collection = read_synonyms_collection(
        filepath=common.SynonymsCollectionFilepathArg.read_argument(args))
    annot_algo = PairBasedOpinionAnnotationAlgorithm(
        dist_in_terms_bound=None, label_provider=ConstantLabelProvider(label_instance=NoLabel()))
    terms_per_context = common.TermsPerContextArg.read_argument(args)
    entities_parser = common.EntitiesParserArg.read_argument(args)
    frames_collection = common.FramesColectionArg.read_argument(args)
    doc_ops = InMemoryDocOperations(docs=input_to_docs(input_texts))
    stemmer = common.StemmerArg.read_argument(args)

    ctx = NetworkSerializationContext(
        labels_scaler=create_labels_scaler(3),
        pos_tagger=POSMystemWrapper(MystemWrapper().MystemInstance),
        frame_roles_label_scaler=ThreeLabelScaler(),
        frames_connotation_provider=RuSentiFramesConnotationProvider(frames_collection))

    embedding = load_embedding_news_mystem_skipgram_1000_20_2015(stemmer)
    bpe_vectorizer = BPEVectorizer(embedding=embedding, max_part_size=3)
    norm_vectorizer = RandomNormalVectorizer(vector_size=embedding.VectorSize,
                                             token_offset=12345)

    pipeline = BasePipeline([
        NetworksInputSerializerPipelineItem(
            vectorizers={
                TermTypes.WORD: bpe_vectorizer,
                TermTypes.ENTITY: bpe_vectorizer,
                TermTypes.FRAME: bpe_vectorizer,
                TermTypes.TOKEN: norm_vectorizer
            },
            str_entity_fmt=create_entity_formatter(common.EntityFormatterTypesArg.read_argument(args)),
            ctx=ctx,
            samples_io=SamplesIO(target_dir=const.OUTPUT_DIR),
            emb_io=NpEmbeddingIO(target_dir=const.OUTPUT_DIR),
            save_labels_func=lambda data_type: data_type != DataType.Test,
            balance_func=lambda data_type: data_type == DataType.Train,
            save_embedding=True)
    ])

    # Initialize text parser with the related dependencies.
    frame_variants_collection = create_and_fill_variant_collection(frames_collection)
    text_parser = BaseTextParser(pipeline=[
        TermsSplitterParser(),
        entities_parser,
        EntitiesGroupingPipelineItem(
            lambda value: SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms_collection, value)),
        DefaultTextTokenizer(keep_tokens=True),
        FrameVariantsParser(frame_variants=frame_variants_collection),
        LemmasBasedFrameVariantsParser(save_lemmas=False,
                                       stemmer=stemmer,
                                       frame_variants=frame_variants_collection),
        FrameVariantsSentimentNegation()])

    # Initialize data processing pipeline.
    test_pipeline = create_neutral_annotation_pipeline(synonyms=synonyms_collection,
                                                       dist_in_terms_bound=terms_per_context,
                                                       terms_per_context=terms_per_context,
                                                       doc_ops=doc_ops,
                                                       text_parser=text_parser,
                                                       dist_in_sentences=0)

    pipeline.run(input_data=None,
                 params_dict={
                     "data_folding": NoFolding(doc_ids=[0], supported_data_type=DataType.Test),
                     "data_type_pipelines": {DataType.Test: test_pipeline}
                 })

from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.pipelines.text_opinion.annot.algo_based import AlgorithmBasedTextOpinionAnnotator
from arekit.contrib.utils.pipelines.text_opinion.extraction import text_opinion_extraction_pipeline
from arekit.contrib.utils.pipelines.text_opinion.filters.distance_based import DistanceLimitedTextOpinionFilter


def create_neutral_annotation_pipeline(synonyms, dist_in_terms_bound, terms_per_context,
                                       doc_ops, text_parser, dist_in_sentences=0):

    nolabel_annotator = AlgorithmBasedTextOpinionAnnotator(
        value_to_group_id_func=lambda value:
            SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
                synonyms=synonyms, value=value),
        annot_algo=PairBasedOpinionAnnotationAlgorithm(
            dist_in_sents=dist_in_sentences,
            dist_in_terms_bound=dist_in_terms_bound,
            label_provider=ConstantLabelProvider(NoLabel())),
        create_empty_collection_func=lambda: OpinionCollection(
            opinions=[],
            synonyms=synonyms,
            error_on_duplicates=True,
            error_on_synonym_end_missed=False))

    annotation_pipeline = text_opinion_extraction_pipeline(
        text_parser=text_parser,
        get_doc_by_id_func=doc_ops.by_id,
        annotators=[
            nolabel_annotator
        ],
        text_opinion_filters=[
            DistanceLimitedTextOpinionFilter(terms_per_context)
        ])

    return annotation_pipeline

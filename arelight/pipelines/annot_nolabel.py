from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.opinions.annot.algo.pair_based import PairBasedOpinionAnnotationAlgorithm
from arekit.common.opinions.annot.algo_based import AlgorithmBasedOpinionAnnotator
from arekit.common.opinions.collection import OpinionCollection
from arekit.common.synonyms.grouping import SynonymsCollectionValuesGroupingProviders
from arekit.contrib.utils.pipelines.annot.base import attitude_extraction_default_pipeline


def create_neutral_annotation_pipeline(synonyms, dist_in_terms_bound, terms_per_context,
                                       doc_ops, text_parser, dist_in_sentences=0):

    annotator = AlgorithmBasedOpinionAnnotator(
        annot_algo=PairBasedOpinionAnnotationAlgorithm(
            dist_in_sents=dist_in_sentences,
            dist_in_terms_bound=dist_in_terms_bound,
            label_provider=ConstantLabelProvider(NoLabel())),
        create_empty_collection_func=lambda: OpinionCollection(
            opinions=[],
            synonyms=synonyms,
            error_on_duplicates=True,
            error_on_synonym_end_missed=False),
        get_doc_existed_opinions_func=lambda _: None)

    annotation_pipeline = attitude_extraction_default_pipeline(
        annotator=annotator,
        get_doc_func=lambda doc_id: doc_ops.get_doc(doc_id),
        text_parser=text_parser,
        value_to_group_id_func=lambda value:
        SynonymsCollectionValuesGroupingProviders.provide_existed_or_register_missed_value(
            synonyms=synonyms, value=value),
        terms_per_context=terms_per_context,
        entity_index_func=None)

    return annotation_pipeline

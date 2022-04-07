import argparse
import sys

sys.path.append('../')

from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.engine import ExperimentEngine
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.types import FoldingType
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.contrib.bert.handlers.serializer import BertExperimentInputSerializerIterationHandler
from arekit.contrib.bert.samplers.types import BertSampleProviderTypes
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from arekit.contrib.experiment_rusentrel.factory import create_experiment
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentNeutralLabel, ExperimentPositiveLabel, \
    ExperimentNegativeLabel
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.experiment_rusentrel.types import ExperimentTypes
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions

from examples.rusentrel.common import Common
from examples.rusentrel.exp_io import CustomRuSentRelNetworkExperimentIO
from network.args import const
from network.args.common import TermsPerContextArg, SynonymsCollectionArg, EntitiesParserArg, InputTextArg, \
    FromFilesArg, RusVectoresEmbeddingFilepathArg, EntityFormatterTypesArg, UseBalancingArg, \
    DistanceInTermsBetweenAttitudeEndsArg, StemmerArg
from network.args.const import DEFAULT_TEXT_FILEPATH
from network.bert.ctx import BertSerializationContext
from utils import create_labels_scaler


class ExperimentBERTTextBThreeScaleLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        stol = {'neu': ExperimentNeutralLabel,
                'pos': ExperimentPositiveLabel,
                'neg': ExperimentNegativeLabel}
        super(ExperimentBERTTextBThreeScaleLabelsFormatter, self).__init__(stol=stol)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Serialization script for obtaining sources, "
                                                 "required for inference and training.")

    # Provide arguments.
    InputTextArg.add_argument(parser, default=None)
    FromFilesArg.add_argument(parser, default=[DEFAULT_TEXT_FILEPATH])
    EntitiesParserArg.add_argument(parser, default="bert-ontonotes")
    RusVectoresEmbeddingFilepathArg.add_argument(parser, default=const.EMBEDDING_FILEPATH)
    TermsPerContextArg.add_argument(parser, default=const.TERMS_PER_CONTEXT)
    SynonymsCollectionArg.add_argument(parser, default=None)
    UseBalancingArg.add_argument(parser, default=True)
    DistanceInTermsBetweenAttitudeEndsArg.add_argument(parser, default=None)
    EntityFormatterTypesArg.add_argument(parser, default="hidden-bert-styled")
    StemmerArg.add_argument(parser, default="mystem")

    # Parsing arguments.
    args = parser.parse_args()

    # Reading arguments.
    text_from_arg = InputTextArg.read_argument(args)
    texts_from_files = FromFilesArg.read_argument(args)
    terms_per_context = TermsPerContextArg.read_argument(args)
    use_balancing = UseBalancingArg.read_argument(args)
    stemmer = StemmerArg.read_argument(args)
    entity_fmt = EntityFormatterTypesArg.read_argument(args)
    dist_in_terms_between_attitude_ends = DistanceInTermsBetweenAttitudeEndsArg.read_argument(args)

    # Predefined parameters.
    labels_count = 3
    rusentrel_version = RuSentRelVersions.V11
    synonyms = RuSentRelSynonymsCollectionProvider.load_collection(stemmer=stemmer)
    folding_type = FoldingType.Fixed

    annot_algo = PairBasedAnnotationAlgorithm(
        dist_in_terms_bound=None,
        label_provider=ConstantLabelProvider(label_instance=ExperimentNeutralLabel()))

    exp_name = Common.create_exp_name(rusentrel_version=rusentrel_version,
                                      ra_version=None,
                                      folding_type=folding_type)

    extra_name_suffix = Common.create_exp_name_suffix(
        use_balancing=use_balancing,
        terms_per_context=terms_per_context,
        dist_in_terms_between_att_ends=dist_in_terms_between_attitude_ends)

    data_folding = Common.create_folding(
        rusentrel_version=rusentrel_version,
        ruattitudes_version=None,
        doc_id_func=lambda doc_id: Common.ra_doc_id_func(doc_id=doc_id))

    # Preparing necessary structures for further initializations.
    exp_ctx = BertSerializationContext(
        label_scaler=create_labels_scaler(labels_count),
        terms_per_context=terms_per_context,
        str_entity_formatter=create_entity_formatter(entity_fmt),
        annotator=DefaultAnnotator(annot_algo=annot_algo),
        name_provider=ExperimentNameProvider(name=exp_name, suffix=extra_name_suffix + "-bert"),
        data_folding=data_folding)

    experiment = create_experiment(
        exp_type=ExperimentTypes.RuSentRel,
        exp_ctx=exp_ctx,
        exp_io=CustomRuSentRelNetworkExperimentIO(exp_ctx),
        folding_type=folding_type,
        rusentrel_version=rusentrel_version,
        ruattitudes_version=None,
        load_ruattitude_docs=True,
        ra_doc_id_func=lambda doc_id: Common.ra_doc_id_func(doc_id=doc_id))

    handler = BertExperimentInputSerializerIterationHandler(
        exp_io=experiment.ExperimentIO,
        exp_ctx=experiment.ExperimentContext,
        doc_ops=experiment.DocumentOperations,
        opin_ops=experiment.OpinionOperations,
        sample_labels_fmt=ExperimentBERTTextBThreeScaleLabelsFormatter(),
        annot_labels_fmt=experiment.OpinionOperations.LabelsFormatter,
        sample_provider_type=BertSampleProviderTypes.NLI_M,
        entity_formatter=experiment.ExperimentContext.StringEntityFormatter,
        value_to_group_id_func=synonyms.get_synonym_group_index,
        balance_train_samples=use_balancing)

    engine = ExperimentEngine(exp_ctx.DataFolding)

    engine.run(handlers=[handler])

from os.path import join, dirname, realpath

from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from arekit.contrib.experiment_rusentrel.entities.types import EntityFormatterTypes
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.experiment_rusentrel.synonyms.provider import RuSentRelSynonymsCollectionProvider
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.source.rusentrel.io_utils import RuSentRelVersions
from arekit.processing.lemmatization.mystem import MystemWrapper

from arelight.pipelines.backend import BratBackendPipelineItem
from arelight.pipelines.inference_bert import BertInferencePipelineItem
from arelight.pipelines.serialize_bert import BertTextsSerializationPipelineItem

from examples.args import const
from examples.utils import create_labels_scaler

current_dir = dirname(realpath(__file__))


def infer_texts_bert(text):
    assert(isinstance(text, str))

    synonyms = RuSentRelSynonymsCollectionProvider.load_collection(
        stemmer=MystemWrapper(),
        version=RuSentRelVersions.V11,
        is_read_only=False)

    labels_scaler = create_labels_scaler(3)

    ppl = BasePipeline(pipeline=[

            BertTextsSerializationPipelineItem(
                synonyms=synonyms,
                terms_per_context=const.TERMS_PER_CONTEXT,
                entities_parser=const.DEFAULT_ENTITIES_PARSER,
                entity_fmt=create_entity_formatter(EntityFormatterTypes.HiddenBertStyled),
                name_provider=ExperimentNameProvider(name="example-bert", suffix="infer"),
                text_b_type="nli_m",
                opin_annot=DefaultAnnotator(
                    PairBasedAnnotationAlgorithm(
                        dist_in_terms_bound=None,
                        label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
                data_folding=NoFolding(doc_ids_to_fold=[0], supported_data_types=[DataType.Test])),

            BertInferencePipelineItem(
                data_type=DataType.Test,
                predict_writer=TsvPredictWriter(),
                bert_config_file=const.BERT_CONFIG_PATH,
                model_checkpoint_path=const.BERT_CKPT_PATH,
                vocab_filepath=const.BERT_VOCAB_PATH,
                max_seq_length=128,
                do_lowercase=const.BERT_DO_LOWERCASE,
                labels_scaler=labels_scaler),

            BratBackendPipelineItem(label_to_rel={
                    str(labels_scaler.label_to_uint(ExperimentPositiveLabel())): "POS",
                    str(labels_scaler.label_to_uint(ExperimentNegativeLabel())): "NEG"
                },
                obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
                rel_color_types={"POS": "GREEN", "NEG": "RED"},
            )
        ])

    filled_template = ppl.run(text, {
        "template_filepath": join(current_dir, "index.tmpl"),
        "predict_fp": None,
        "brat_vis_fp": None
    })

    return filled_template

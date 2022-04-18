from arekit.common.experiment.annot.algo.pair_based import PairBasedAnnotationAlgorithm
from arekit.common.experiment.annot.default import DefaultAnnotator
from arekit.common.experiment.data_type import DataType
from arekit.common.experiment.name_provider import ExperimentNameProvider
from arekit.common.folding.nofold import NoFolding
from arekit.common.labels.base import NoLabel
from arekit.common.labels.provider.constant import ConstantLabelProvider
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.bert.samplers.types import SampleFormattersService
from arekit.contrib.experiment_rusentrel.entities.factory import create_entity_formatter
from arekit.contrib.experiment_rusentrel.entities.types import EntityFormatterTypes
from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.types import ExperimentPositiveLabel, ExperimentNegativeLabel
from arekit.contrib.experiment_rusentrel.synonyms.collection import StemmerBasedSynonymCollection
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups
from arekit.processing.lemmatization.mystem import MystemWrapper

from arelight.pipelines.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.inference_bert import BertInferencePipelineItem
from arelight.pipelines.serialize_bert import BertTextsSerializationPipelineItem
from arelight.text.pipeline_entities_bert_ontonotes import BertOntonotesNERPipelineItem


def iter_groups(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        for group in iter_synonym_groups(file):
            yield group


def demo_infer_texts_bert_pipeline(texts_count,
                                   synonyms_filepath,
                                   output_dir,
                                   bert_config_path,
                                   bert_vocab_path,
                                   bert_finetuned_ckpt_path,
                                   text_b_type=SampleFormattersService.name_to_type("nli_m"),
                                   labels_scaler=ThreeLabelScaler(),
                                   terms_per_context=50,
                                   do_lowercase=False,
                                   max_seq_length=128):
    assert(isinstance(texts_count, int))
    assert(isinstance(output_dir, str))
    assert(isinstance(synonyms_filepath, str))

    synonyms = StemmerBasedSynonymCollection(
        iter_group_values_lists=iter_groups(synonyms_filepath),
        stemmer=MystemWrapper(),
        is_read_only=False, debug=False)

    ppl = BasePipeline(pipeline=[

        BertTextsSerializationPipelineItem(
            synonyms=synonyms,
            terms_per_context=terms_per_context,
            entities_parser=BertOntonotesNERPipelineItem(
                lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"]),
            entity_fmt=create_entity_formatter(EntityFormatterTypes.HiddenBertStyled),
            name_provider=ExperimentNameProvider(name="example-bert", suffix="infer"),
            text_b_type=text_b_type,
            output_dir=output_dir,
            opin_annot=DefaultAnnotator(
                PairBasedAnnotationAlgorithm(
                    dist_in_terms_bound=None,
                    label_provider=ConstantLabelProvider(label_instance=NoLabel()))),
            data_folding=NoFolding(doc_ids_to_fold=list(range(texts_count)),
                                   supported_data_types=[DataType.Test])),

        BertInferencePipelineItem(
            data_type=DataType.Test,
            predict_writer=TsvPredictWriter(),
            bert_config_file=bert_config_path,
            model_checkpoint_path=bert_finetuned_ckpt_path,
            vocab_filepath=bert_vocab_path,
            max_seq_length=max_seq_length,
            do_lowercase=do_lowercase,
            labels_scaler=labels_scaler),

        BratBackendContentsPipelineItem(label_to_rel={
            str(labels_scaler.label_to_uint(ExperimentPositiveLabel())): "POS",
            str(labels_scaler.label_to_uint(ExperimentNegativeLabel())): "NEG"
        },
            obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN", "NEG": "RED"},
        )
    ])

    return ppl

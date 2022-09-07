from arekit.common.data.input.readers.tsv import TsvReader
from arekit.common.data.input.writers.tsv import TsvWriter
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.pipeline.base import BasePipeline
from arekit.contrib.networks.core.predict.tsv_writer import TsvPredictWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.bert import BertExperimentInputSerializerPipelineItem

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.items.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.items.inference_bert import BertInferencePipelineItem
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import SampleFormattersService


def demo_infer_texts_bert_pipeline(texts_count,
                                   output_dir,
                                   bert_config_path,
                                   bert_vocab_path,
                                   bert_finetuned_ckpt_path,
                                   entity_fmt,
                                   labels_scaler,
                                   text_b_type=SampleFormattersService.name_to_type("nli_m"),
                                   do_lowercase=False,
                                   max_seq_length=128):
    assert(isinstance(texts_count, int))
    assert(isinstance(output_dir, str))
    assert(isinstance(labels_scaler, BaseLabelScaler))

    samples_io = SamplesIO(target_dir=output_dir, reader=TsvReader(), writer=TsvWriter(write_header=True))

    pipeline = BasePipeline(pipeline=[

        BertExperimentInputSerializerPipelineItem(
            sample_rows_provider=create_bert_sample_provider(
                provider_type=text_b_type,
                label_scaler=labels_scaler,
                entity_formatter=entity_fmt),
            samples_io=samples_io,
            save_labels_func=lambda data_type: data_type != DataType.Test,
            balance_func=lambda data_type: data_type == DataType.Train),

        BertInferencePipelineItem(
            data_type=DataType.Test,
            samples_io=samples_io,
            predict_writer=TsvPredictWriter(),
            bert_config_file=bert_config_path,
            model_checkpoint_path=bert_finetuned_ckpt_path,
            vocab_filepath=bert_vocab_path,
            max_seq_length=max_seq_length,
            do_lowercase=do_lowercase,
            labels_scaler=labels_scaler),

        BratBackendContentsPipelineItem(label_to_rel={
            str(labels_scaler.label_to_uint(PositiveLabel())): "POS",
            str(labels_scaler.label_to_uint(NegativeLabel())): "NEG"
        },
            obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN", "NEG": "RED"},
        )
    ])

    return pipeline

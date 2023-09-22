from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.contrib.bert.input.providers.text_pair import PairTextProvider
from arekit.contrib.utils.data.readers.csv_pd import PandasCsvReader
from arekit.contrib.utils.data.storages.row_cache import RowCacheStorage
from arekit.contrib.utils.data.writers.csv_native import NativeCsvWriter
from arekit.contrib.utils.data.writers.json_opennre import OpenNREJsonWriter
from arekit.contrib.utils.io_utils.samples import SamplesIO
from arekit.contrib.utils.pipelines.items.sampling.base import BaseSerializerPipelineItem

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.items.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.items.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.items.inference_bert import BertInferencePipelineItem
from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.predict_writer_csv import TsvPredictWriter
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import SampleFormattersService


def demo_infer_texts_bert_pipeline(pretrained_bert,
                                   samples_output_dir,
                                   samples_prefix,
                                   entity_fmt,
                                   labels_scaler,
                                   checkpoint_path,
                                   bert_type,
                                   bert_config_path=None,
                                   bert_vocab_path=None,
                                   brat_backend=False,
                                   text_b_type=SampleFormattersService.name_to_type("nli_m"),
                                   max_seq_length=128):
    assert(isinstance(bert_type, str))
    assert(isinstance(pretrained_bert, str) or pretrained_bert is None)
    assert(isinstance(samples_output_dir, str))
    assert(isinstance(samples_prefix, str))
    assert(isinstance(labels_scaler, BaseLabelScaler))

    # Setup Writer.
    if bert_type == "opennre":
        # OpenNRE supports the specific type of writer based on JSONL.
        writer = OpenNREJsonWriter(text_columns=[BaseSingleTextProvider.TEXT_A, PairTextProvider.TEXT_B],
                                   keep_extra_columns=False,
                                   # `0` basically.
                                   na_value=str(labels_scaler.label_to_uint(NoLabel())))
    else:
        writer = NativeCsvWriter(delimiter=',')

    # Setup SamplesIO.
    samples_io = SamplesIO(target_dir=samples_output_dir,
                           reader=PandasCsvReader(sep=',', compression="infer"),
                           prefix=samples_prefix,
                           writer=writer)

    # Serialization by default in the pipeline.
    pipeline = [
        BaseSerializerPipelineItem(
            rows_provider=create_bert_sample_provider(
                provider_type=text_b_type,
                label_scaler=labels_scaler,
                entity_formatter=entity_fmt),
            samples_io=samples_io,
            storage=RowCacheStorage(force_collect_columns=[
                # These additional columns required for BRAT visualization.
                const.ENTITIES, const.ENTITY_VALUES, const.ENTITY_TYPES, const.SENT_IND
            ]),
            save_labels_func=lambda data_type: data_type != DataType.Test)
    ]

    if pretrained_bert is None:
        return pipeline

    # Add BERT processing pipeline.
    if bert_type == "deeppavlov":
        pipeline += [
            BertInferencePipelineItem(
                pretrained_bert=pretrained_bert,
                data_type=DataType.Test,
                samples_io=samples_io,
                predict_writer=TsvPredictWriter(),
                bert_config_file=bert_config_path,
                vocab_filepath=bert_vocab_path,
                max_seq_length=max_seq_length,
                labels_count=labels_scaler.LabelsCount),
        ]
    elif bert_type == "opennre":
        pipeline += [
            BertOpenNREInferencePipelineItem(
                pretrained_bert=pretrained_bert,
                data_type=DataType.Test,
                samples_io=samples_io,
                labels_scaler=labels_scaler,
                checkpoint_path=checkpoint_path,
                max_seq_length=max_seq_length,
                predict_writer=TsvPredictWriter(),
                batch_size=10,
            )
        ]
    else:
        raise Exception("Not supported bert_type: {}".format(bert_type))

    if not brat_backend:
        return pipeline

    pipeline += [
        BratBackendContentsPipelineItem(label_to_rel={
            str(labels_scaler.label_to_uint(PositiveLabel())): "POS",
            str(labels_scaler.label_to_uint(NegativeLabel())): "NEG"
        },
            obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
            rel_color_types={"POS": "GREEN", "NEG": "RED"},
        )
    ]

    pipeline += [
        BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/")
    ]

    return pipeline

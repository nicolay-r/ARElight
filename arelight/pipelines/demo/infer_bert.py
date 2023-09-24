from arekit.common.data import const
from arekit.common.data.input.providers.text.single import BaseSingleTextProvider
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.base import NoLabel
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
from arelight.pipelines.items.backend_d3js_graphs import D3jsGraphsBackendPipelineItem
from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.pipelines.items.inference_transformers_dp import TransformersDeepPavlovInferencePipelineItem
from arelight.predict_writer_csv import TsvPredictWriter
from arelight.samplers.bert import create_bert_sample_provider
from arelight.samplers.types import SampleFormattersService


def demo_infer_texts_bert_pipeline(sampling_engines="arekit", infer_engines=None, backend_engines=None,
                                   # TODO. These parameters below are expected to be a part of the
                                   # TODO. Input options (ParlAI alike) structure and passed
                                   # TODO. as a single opt parameter.
                                   pretrained_bert=None, samples_output_dir=None, samples_prefix="samples",
                                   entity_fmt=None, labels_scaler=None, checkpoint_path=None,
                                   text_b_type=SampleFormattersService.name_to_type("nli_m"),
                                   max_seq_length=128):
    assert(isinstance(infer_engines, list) or infer_engines is None or isinstance(infer_engines, str))
    assert(isinstance(sampling_engines, list) or sampling_engines is None or isinstance(sampling_engines, str))
    assert(isinstance(backend_engines, list) or backend_engines is None or isinstance(backend_engines, str))

    infer_engines = [infer_engines] if isinstance(infer_engines, str) else infer_engines
    infer_engines = [] if infer_engines is None else infer_engines

    sampling_engines = [sampling_engines] if isinstance(sampling_engines, str) else sampling_engines
    sampling_engines = [] if sampling_engines is None else sampling_engines

    backend_engines = [backend_engines] if isinstance(backend_engines, str) else backend_engines
    backend_engines = [] if backend_engines is None else backend_engines

    #####################################################################
    # Setup Common Parameters.
    #####################################################################

    if "opennre" in infer_engines:
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

    pipeline = []

    #####################################################################
    # Serialization Items
    #####################################################################

    if "arekit" in sampling_engines:
        pipeline += [
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

    #####################################################################
    # Inference Items
    #####################################################################

    if "deeppavlov" in infer_engines:
        pipeline += [
            TransformersDeepPavlovInferencePipelineItem(
                pretrained_bert=pretrained_bert,
                data_type=DataType.Test,
                samples_io=samples_io,
                predict_writer=TsvPredictWriter(),
                max_seq_length=max_seq_length,
                labels_count=labels_scaler.LabelsCount)
        ]

    if "opennre" in infer_engines:
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

    #####################################################################
    # Backend Items (after inference)
    #####################################################################

    if "brat" in backend_engines:
        pipeline += [
            BratBackendContentsPipelineItem(label_to_rel={
                str(labels_scaler.label_to_uint(PositiveLabel())): "POS",
                str(labels_scaler.label_to_uint(NegativeLabel())): "NEG"
            },
                obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
                rel_color_types={"POS": "GREEN", "NEG": "RED"},
            ),
            BratHtmlEmbeddingPipelineItem(brat_url="http://localhost:8001/")
        ]

    if "d3js_graphs" in backend_engines:
        pipeline += [
            D3jsGraphsBackendPipelineItem(samples_io)
        ]

    return pipeline

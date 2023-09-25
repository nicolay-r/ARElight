from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.items.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.items.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.items.backend_d3js_graphs import D3jsGraphsBackendPipelineItem
from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.pipelines.items.inference_transformers_dp import TransformersDeepPavlovInferencePipelineItem
from arelight.pipelines.items.serializer_arekit import AREkitSerializerPipelineItem


def demo_infer_texts_bert_pipeline(sampling_engines="arekit", infer_engines=None, backend_engines=None,
                                   pretrained_bert=None, labels_scaler=None, checkpoint_path=None,
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

    pipeline = []
    #####################################################################
    # Serialization Items
    #####################################################################

    if "arekit" in sampling_engines:
        pipeline += [AREkitSerializerPipelineItem()]

    #####################################################################
    # Inference Items
    #####################################################################

    if "deeppavlov" in infer_engines:
        pipeline += [
            TransformersDeepPavlovInferencePipelineItem(
                pretrained_bert=pretrained_bert,
                max_seq_length=max_seq_length,
                labels_count=labels_scaler.LabelsCount)
        ]

    if "opennre" in infer_engines:
        pipeline += [
            BertOpenNREInferencePipelineItem(
                pretrained_bert=pretrained_bert,
                labels_scaler=labels_scaler,
                checkpoint_path=checkpoint_path,
                max_seq_length=max_seq_length,
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
            BratHtmlEmbeddingPipelineItem()
        ]

    if "d3js_graphs" in backend_engines:
        pipeline += [
            D3jsGraphsBackendPipelineItem()
        ]

    return pipeline

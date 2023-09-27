from arelight.pipelines.items.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.items.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.items.backend_d3js_graphs import D3jsGraphsBackendPipelineItem
from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.pipelines.items.inference_transformers_dp import TransformersDeepPavlovInferencePipelineItem
from arelight.pipelines.items.inference_writer import InferenceWriterPipelineItem
from arelight.pipelines.items.serializer_arekit import AREkitSerializerPipelineItem
from arelight.predict_writer_csv import TsvPredictWriter


def demo_infer_texts_bert_pipeline(sampling_engines="arekit", infer_engines=None, backend_engines=None):
    assert(isinstance(sampling_engines, list) or sampling_engines is None or isinstance(sampling_engines, str))
    assert(isinstance(infer_engines, dict))
    assert(isinstance(backend_engines, dict))

    sampling_engines = [sampling_engines] if isinstance(sampling_engines, str) else sampling_engines
    sampling_engines = [] if sampling_engines is None else sampling_engines

    pipeline = []
    #####################################################################
    # Serialization Items
    #####################################################################

    if "arekit" in sampling_engines:
        pipeline += [AREkitSerializerPipelineItem()]

    #####################################################################
    # Inference Items
    #####################################################################

    inference_writer = TsvPredictWriter()

    if "deeppavlov" in infer_engines:
        pipeline += [TransformersDeepPavlovInferencePipelineItem(**infer_engines["deeppavlov"]),
                     InferenceWriterPipelineItem(inference_writer)]

    if "opennre" in infer_engines:
        pipeline += [BertOpenNREInferencePipelineItem(**infer_engines["opennre"]),
                     InferenceWriterPipelineItem(inference_writer)]

    #####################################################################
    # Backend Items (after inference)
    #####################################################################

    if "brat" in backend_engines:
        pipeline += [
            BratBackendContentsPipelineItem(
                obj_color_types={"ORG": '#7fa2ff', "GPE": "#7fa200", "PERSON": "#7f00ff", "Frame": "#00a2ff"},
                rel_color_types={"POS": "GREEN", "NEG": "RED"}
            ),
            BratHtmlEmbeddingPipelineItem()
        ]

    if "d3js_graphs" in backend_engines:
        pipeline += [
            D3jsGraphsBackendPipelineItem(**backend_engines["d3js_graphs"])
        ]

    return pipeline

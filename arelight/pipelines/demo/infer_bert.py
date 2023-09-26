from arelight.pipelines.items.backend_brat_html import BratHtmlEmbeddingPipelineItem
from arelight.pipelines.items.backend_brat_json import BratBackendContentsPipelineItem
from arelight.pipelines.items.backend_d3js_graphs import D3jsGraphsBackendPipelineItem
from arelight.pipelines.items.inference_bert_opennre import BertOpenNREInferencePipelineItem
from arelight.pipelines.items.inference_transformers_dp import TransformersDeepPavlovInferencePipelineItem
from arelight.pipelines.items.inference_writer import InferenceWriterPipelineItem
from arelight.pipelines.items.serializer_arekit import AREkitSerializerPipelineItem


def demo_infer_texts_bert_pipeline(sampling_engines="arekit", infer_engines=None, backend_engines=None):
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
        pipeline += [TransformersDeepPavlovInferencePipelineItem(),
                     InferenceWriterPipelineItem()]

    if "opennre" in infer_engines:
        pipeline += [BertOpenNREInferencePipelineItem(),
                     InferenceWriterPipelineItem()]

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
            D3jsGraphsBackendPipelineItem()
        ]

    return pipeline

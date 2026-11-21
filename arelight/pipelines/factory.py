from arelight.const import BULK_CHAIN
from arelight.pipelines.items.inference_writer import InferenceWriterPipelineItem


def build_pipeline(sampling_engines=None, infer_engines=None, inference_writer=None):
    assert(isinstance(sampling_engines, dict) or sampling_engines is None)
    assert(isinstance(infer_engines, dict) or infer_engines is None)

    sampling_engines = {} if sampling_engines is None else sampling_engines
    infer_engines = {} if infer_engines is None else infer_engines

    pipeline = []
    #####################################################################
    # Serialization Items
    #####################################################################
    if "arekit" in sampling_engines:
        from arelight.pipelines.items.serializer_arekit import AREkitSerializerPipelineItem
        pipeline += [AREkitSerializerPipelineItem(**sampling_engines["arekit"])]

    #####################################################################
    # Inference Items
    #####################################################################
    if BULK_CHAIN in infer_engines:
        from arelight.pipelines.items.inference_bulkchain import InferenceBulkChainPipelineItem
        pipeline += [InferenceBulkChainPipelineItem(**infer_engines[BULK_CHAIN]),
                     InferenceWriterPipelineItem(inference_writer)]

    return pipeline

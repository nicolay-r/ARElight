from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer import BasePredictWriter


class InferenceWriterPipelineItem(BasePipelineItem):

    def __init__(self, writer):
        assert(isinstance(writer, BasePredictWriter))
        self.__writer = writer

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))

        # Setup predicted result writer.
        target = input_data.provide("predict_filepath")
        print(target)

        self.__writer.set_target(target)

        # Gathering the content
        title, contents_it = BasePredictProvider().provide(
            sample_id_with_uint_labels_iter=input_data.provide("iter_infer"),
            labels_count=input_data.provide("labels_scaler").LabelsCount)

        with self.__writer:
            self.__writer.write(title=title, contents_it=contents_it,
                                total=input_data.provide_or_none("iter_total"))

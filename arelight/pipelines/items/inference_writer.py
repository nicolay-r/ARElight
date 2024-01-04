from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer import BasePredictWriter


class InferenceWriterPipelineItem(BasePipelineItem):

    def __init__(self, writer, **kwargs):
        assert(isinstance(writer, BasePredictWriter))
        super(InferenceWriterPipelineItem, self).__init__(**kwargs)
        self.__writer = writer

    def apply_core(self, input_data, pipeline_ctx):

        # Setup predicted result writer.
        target = pipeline_ctx.provide("predict_filepath")

        self.__writer.set_target(target)

        # Gathering the content
        title, contents_it = BasePredictProvider().provide(
            sample_id_with_uint_labels_iter=pipeline_ctx.provide("iter_infer"),
            labels_count=pipeline_ctx.provide("labels_scaler").LabelsCount)

        with self.__writer:
            self.__writer.write(title=title, contents_it=contents_it,
                                total=pipeline_ctx.provide_or_none("iter_total"))

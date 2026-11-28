from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.predict.header import PredictHeader
from arelight.predict.writer import BasePredictWriter


class InferenceWriterPipelineItem(BasePipelineItem):

    def __init__(self, writer, provider, **kwargs):
        assert(isinstance(writer, BasePredictWriter))
        super(InferenceWriterPipelineItem, self).__init__(**kwargs)
        self.__writer = writer
        self.__provider = provider

    def apply_core(self, input_data, pipeline_ctx):

        # Setup predicted result writer.
        target = pipeline_ctx.provide("predict_filepath")

        self.__writer.set_target(target)

        # Extracting list of the uint labels.
        labels_scaler = pipeline_ctx.provide("labels_scaler")
        uint_labels = [labels_scaler.label_to_uint(label) for label in labels_scaler.ordered_suppoted_labels()]

        # Gathering the content.
        contents_it = self.__provider.handle_iter(
            data_it=pipeline_ctx.provide("iter_infer"),
            uint_labels=uint_labels)
        header = PredictHeader.create_header(uint_labels=uint_labels)

        with self.__writer:
            self.__writer.write(header=header, contents_it=contents_it, total=None)

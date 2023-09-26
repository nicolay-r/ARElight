from os.path import dirname, join

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.predict_provider import BasePredictProvider
from arelight.predict_writer import BasePredictWriter


class InferenceWriterPipelineItem(BasePipelineItem):

    def __init__(self, writer):
        assert(isinstance(writer, BasePredictWriter))
        self.__writer = writer

    def apply_core(self, input_data, pipeline_ctx):

        # Fetch other required in further information from input_data.
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)

        # Setup predicted result writer.
        tgt = input_data.provide_or_none("predict_filepath")
        if tgt is None:
            tgt = join(dirname(samples_filepath), "predict.tsv.gz")

        self.__writer.set_target(tgt)

        # Update for further pipeline items.
        input_data.update("predict_filepath", tgt)

        # Gathering the content
        title, contents_it = BasePredictProvider().provide(
            sample_id_with_uint_labels_iter=input_data.provide("iter_infer"),
            labels_count=input_data.provide("labels_scaler").LabelsCount)

        with self.__writer:
            self.__writer.write(title=title, contents_it=contents_it)

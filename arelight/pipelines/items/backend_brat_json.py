from os.path import dirname

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.backend.brat.converter import BratBackend
from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel
from arelight.pipelines.demo.labels.scalers import ThreeLabelScaler


class BratBackendContentsPipelineItem(BasePipelineItem):

    def __init__(self, obj_color_types, rel_color_types, brat_url="http://localhost:8001/"):
        self.__brat_be = BratBackend()
        self.__brat_url = brat_url
        self.__obj_color_types = obj_color_types
        self.__rel_color_types = rel_color_types

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))
        assert(isinstance(pipeline_ctx, PipelineContext))

        # Obtaining input parameters.
        # NOTE: At present we consider sentiment label scaler.
        labels_scaler = input_data.provide("labels_scaler")
        assert(isinstance(labels_scaler, ThreeLabelScaler))

        predict_filepath = input_data.provide("predict_filepath")

        # Setup labels mapping.
        label_to_rel = {
            str(labels_scaler.label_to_uint(PositiveLabel())): "POS",
            str(labels_scaler.label_to_uint(NegativeLabel())): "NEG"
        }

        # Obtain the samples io.
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)

        contents = self.__brat_be.to_data(
           infer_predict_filepath=predict_filepath,
           samples_data_filepath=samples_filepath,
           obj_color_types=self.__obj_color_types,
           rel_color_types=self.__rel_color_types,
           label_to_rel=label_to_rel)

        exp_root = dirname(samples_filepath)

        pipeline_ctx.update("exp_root", exp_root)

        # Save the result data in the pipeline output.
        for key, value in contents.items():
            input_data.update(key, value)

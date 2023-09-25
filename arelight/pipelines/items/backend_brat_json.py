from os.path import dirname

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.backend.brat.converter import BratBackend


class BratBackendContentsPipelineItem(BasePipelineItem):

    def __init__(self, obj_color_types, rel_color_types,
                 label_to_rel, brat_url="http://localhost:8001/"):
        self.__brat_be = BratBackend()
        self.__brat_url = brat_url
        self.__obj_color_types = obj_color_types
        self.__rel_color_types = rel_color_types
        self.__label_to_rel = label_to_rel

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))
        assert(isinstance(pipeline_ctx, PipelineContext))

        # Obtain the samples io.
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)

        contents = self.__brat_be.to_data(
           infer_predict_filepath=pipeline_ctx.provide("predict_fp"),
           samples_data_filepath=samples_filepath,
           obj_color_types=self.__obj_color_types,
           rel_color_types=self.__rel_color_types,
           label_to_rel=self.__label_to_rel)

        exp_root = dirname(samples_filepath)

        pipeline_ctx.update("exp_root", exp_root)

        # Save the result data in the pipeline output.
        for key, value in contents.items():
            input_data.update(key, value)

import os

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.brat_backend import BratBackend
from arelight.exp.exp_io import InferIOUtils


class BratBackendContentsPipelineItem(BasePipelineItem):

    def __init__(self, obj_color_types, rel_color_types,
                 label_to_rel, brat_url="http://localhost:8001/"):
        self.__brat_be = BratBackend()
        self.__brat_url = brat_url
        self.__obj_color_types = obj_color_types
        self.__rel_color_types = rel_color_types
        self.__label_to_rel = label_to_rel

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, InferIOUtils))
        assert(isinstance(pipeline_ctx, PipelineContext))

        contents = self.__brat_be.to_data(
           result_data_filepath=pipeline_ctx.provide("predict_fp"),
           samples_data_filepath=input_data.create_samples_writer_target(DataType.Test),
           obj_color_types=self.__obj_color_types,
           rel_color_types=self.__rel_color_types,
           label_to_rel=self.__label_to_rel)

        exp_root = os.path.join(input_data._get_experiment_sources_dir(),
                                input_data.get_experiment_folder_name())

        pipeline_ctx.update("exp_root", exp_root)

        return contents

import logging

from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.arekit.api.data_type import DataType
from arelight.utils import init_llm, infer_async_batch_it


class InferenceBulkChainPipelineItem(BasePipelineItem):

    def __init__(self, class_name, model_name, task_kwargs, **kwargs):
        super(InferenceBulkChainPipelineItem, self).__init__(**kwargs)
        self.__model = init_llm(class_filepath=class_name, model_name=model_name)
        self.__task_kwargs = task_kwargs

        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

    def __iter_infer(self, samples):
        data_it = infer_async_batch_it(
            llm=self.__model,
            input_dicts_it=(row for _, row in samples),
            **self.__task_kwargs)

        label_column = self.__task_kwargs["schema"][-1]["out"]
        for row in data_it:
            yield [
                row[self.__task_kwargs['default_id_column']],
                row[label_column],
            ]

    def apply_core(self, input_data, pipeline_ctx):
        samples_io = pipeline_ctx.provide("samples_io")
        samples_filepath = pipeline_ctx.provide_or_none("bulkchain_samples_filepath")
        if samples_filepath is None:
            samples_filepath = samples_io.create_target(data_type=DataType.Test)

        assert(self.__model is not None)

        pipeline_ctx.update("iter_infer", self.__iter_infer(samples_io.Reader.read(samples_filepath)))

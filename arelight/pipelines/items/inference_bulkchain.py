import logging

from arekit.common.experiment.data_type import DataType
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.third_party.sqlite3 import SQLite3Service
from arelight.utils import init_llm, infer_async_batch_it


class InferenceBulkChainPipelineItem(BasePipelineItem):

    def __init__(self, class_name, model_name, api_key, table_name, task_kwargs, **kwargs):
        super(InferenceBulkChainPipelineItem, self).__init__(**kwargs)
        self.__sqlite_service = SQLite3Service()
        self.__table_name = table_name
        self.__model = init_llm(class_filepath=class_name, model_name=model_name, api_key=api_key)
        self.__task_kwargs = task_kwargs

        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("httpcore.http11").setLevel(logging.WARNING)

    def __iter_predict_result(self, samples_filepath):
        self.__sqlite_service.connect(samples_filepath)

        data_it = infer_async_batch_it(
            llm=self.__model,
            input_dicts_it=self.__sqlite_service.iter_rows(table_name=self.__table_name, return_dict=True),
            **self.__task_kwargs)

        for row in data_it:
            yield [
                row[self.__task_kwargs['default_id_column']],
                self.__task_kwargs['class_to_int'](row)
            ]

        self.__sqlite_service.disconnect()

    def __total(self, samples_filepath):
        self.__sqlite_service.connect(samples_filepath)
        return self.__sqlite_service.table_rows_count(table_name=self.__table_name)
        self.__sqlite_service.disconnect()

    def apply_core(self, input_data, pipeline_ctx):
        # Try to obtain from the specific input variable.
        samples_filepath = pipeline_ctx.provide_or_none("bulkchain_samples_filepath")
        if samples_filepath is None:
            samples_io = pipeline_ctx.provide("samples_io")
            samples_filepath = samples_io.create_target(data_type=DataType.Test)

        assert(self.__model is not None)

        pipeline_ctx.update("iter_infer", self.__iter_predict_result(samples_filepath))
        pipeline_ctx.update("iter_total", self.__total(samples_filepath))

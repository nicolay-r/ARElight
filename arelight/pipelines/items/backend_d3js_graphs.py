import logging

from arekit.common.data import const
from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.data.rows_parser import ParsedSampleRow
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array
from arelight.predict.provider import BasePredictProvider

logger = logging.getLogger(__name__)


class D3jsGraphsBackendPipelineItem(BasePipelineItem):

    def __init__(self, graph_min_links=0.01, graph_a_labels=None, weights=True, **kwargs):
        super(D3jsGraphsBackendPipelineItem, self).__init__(**kwargs)
        self.__graph_min_links = graph_min_links

        # Setup filters for the A and B graphs for further operations application.
        self.__graph_label_filter = lambda _: True if graph_a_labels is None else lambda label: label in graph_a_labels

        # Considering weights for graphs.
        self.__graph_weights = weights

        # Parameters for sampler.
        self.__column_fmts = [create_base_column_fmt(fmt_type="parser")]

    @staticmethod
    def __iter_relations(samples, columns_fmts, labels, labels_filter_func=None):
        assert(isinstance(samples, BaseRowsStorage))
        assert(isinstance(labels, list))
        assert(callable(labels_filter_func) or labels_filter_func is None)

        for ind, row_data in enumerate(samples):
            _, sample_row = row_data
            parsed_row = ParsedSampleRow(sample_row, columns_fmts=columns_fmts, no_value_func=lambda: None)
            label = labels[ind]

            if labels_filter_func is not None and not labels_filter_func(label):
                continue

            source = ParsedSampleRowExtraService.calc("SourceValue", parsed_row=parsed_row)
            target = ParsedSampleRowExtraService.calc("TargetValue", parsed_row=parsed_row)
            yield [source, target, label]

    def iter_column_value(self, samples, column_value):
        for _, sample_row in samples:
            parsed_row = ParsedSampleRow(row={column_value: sample_row[column_value]},
                                         columns_fmts=self.__column_fmts,
                                         no_value_func=lambda: None)
            yield parsed_row[column_value]

    def apply_core(self, input_data, pipeline_ctx):

        collection_name = pipeline_ctx.provide("d3js_collection_name")
        predict_filepath = pipeline_ctx.provide("predict_filepath")
        result_reader = pipeline_ctx.provide("predict_reader")
        labels_fmt = pipeline_ctx.provide("labels_formatter")
        assert(isinstance(labels_fmt, StringLabelsFormatter))
        labels_scaler = pipeline_ctx.provide("labels_scaler")
        assert(isinstance(labels_scaler, BaseLabelScaler))
        predict_storage = result_reader.read(predict_filepath)
        assert(isinstance(predict_storage, BaseRowsStorage))

        # Reading samples.
        samples_io = pipeline_ctx.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)
        samples = samples_io.Reader.read(samples_filepath)

        # Reading labels.
        uint_labels_iter = BasePredictProvider.iter_from_storage(
            predict_data=predict_storage,
            uint_labels=[labels_scaler.label_to_uint(label) for label in labels_scaler.ordered_suppoted_labels()],
            keep_ind=False)

        labels = list(map(lambda item: labels_fmt.label_to_str(labels_scaler.uint_to_label(item)), uint_labels_iter))

        graph = make_graph_from_relations_array(
            graph_name=collection_name,
            relations=self.__iter_relations(samples=samples,
                                            labels=labels,
                                            labels_filter_func=self.__graph_label_filter,
                                            columns_fmts=self.__column_fmts),
            entity_values=self.iter_column_value(samples=samples, column_value=const.ENTITY_VALUES),
            entity_types=self.iter_column_value(samples=samples, column_value=const.ENTITY_TYPES),
            min_links=self.__graph_min_links,
            weights=self.__graph_weights)

        # Saving graph as the collection name for it.
        pipeline_ctx.update("d3js_graph_a", value=graph)

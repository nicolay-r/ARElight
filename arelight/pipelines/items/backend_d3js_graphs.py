import logging

from arekit.common.data import const
from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.data.rows_parser import ParsedSampleRow
from arekit.common.data.storages.base import BaseRowsStorage
from arekit.common.experiment.data_type import DataType
from arekit.common.labels.scaler.base import BaseLabelScaler
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.arekit.parse_predict import iter_predicted_labels
from arelight.arekit.parsed_row_service import ParsedSampleRowExtraService
from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array


logger = logging.getLogger(__name__)


class D3jsGraphsBackendPipelineItem(BasePipelineItem):

    def __init__(self, graph_min_links=0.01, graph_a_labels=None, weights=True):
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
        assert(isinstance(input_data, PipelineContext))

        predict_filepath = input_data.provide("predict_filepath")
        result_reader = input_data.provide("predict_reader")
        labels_fmt = input_data.provide("labels_formatter")
        assert(isinstance(labels_fmt, StringLabelsFormatter))
        labels_scaler = input_data.provide("labels_scaler")
        assert(isinstance(labels_scaler, BaseLabelScaler))
        predict_storage = result_reader.read(predict_filepath)
        assert(isinstance(predict_storage, BaseRowsStorage))

        # Reading samples.
        samples_io = input_data.provide("samples_io")
        samples_filepath = samples_io.create_target(data_type=DataType.Test)
        samples = samples_io.Reader.read(samples_filepath)

        # Reading labels.
        labels_to_str = {str(labels_scaler.label_to_uint(label)): labels_fmt.label_to_str(label)
                         for label in labels_scaler.ordered_suppoted_labels()}
        labels = list(iter_predicted_labels(predict_data=predict_storage, label_to_str=labels_to_str, keep_ind=False))

        graph = make_graph_from_relations_array(
            relations=self.__iter_relations(samples=samples,
                                            labels=labels,
                                            labels_filter_func=self.__graph_label_filter,
                                            columns_fmts=self.__column_fmts),
            entity_values=self.iter_column_value(samples=samples, column_value=const.ENTITY_VALUES),
            entity_types=self.iter_column_value(samples=samples, column_value=const.ENTITY_TYPES),
            min_links=self.__graph_min_links,
            weights=self.__graph_weights)

        # Saving graph as the collection name for it.
        input_data.update("d3js_graph_a", value=graph)
        input_data.update("d3js_collection_name", value=samples_io.Prefix)

import logging
import os
from os.path import join

from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.labels.str_fmt import StringLabelsFormatter
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web import iter_ui_backend_folders, GRAPH_TYPE_RADIAL
from arelight.backend.d3js.utils_graph import save_graph


logger = logging.getLogger(__name__)


class D3jsGraphOperationsBackendPipelineItem(BasePipelineItem):

    def __init__(self, **kwargs):
        # Parameters for sampler.
        super(D3jsGraphOperationsBackendPipelineItem, self).__init__(**kwargs)
        self.__column_fmts = [create_base_column_fmt(fmt_type="parser")]

    def apply_core(self, input_data, pipeline_ctx):

        graph_a = pipeline_ctx.provide_or_none("d3js_graph_a")
        graph_b = pipeline_ctx.provide_or_none("d3js_graph_b")
        op = pipeline_ctx.provide_or_none("d3js_graph_operations")
        weights = pipeline_ctx.provide_or_none("d3js_graph_weights")
        target_dir = pipeline_ctx.provide("d3js_graph_output_dir")
        collection_name = pipeline_ctx.provide("d3js_collection_name")
        labels_fmt = pipeline_ctx.provide("labels_formatter")
        assert(isinstance(labels_fmt, StringLabelsFormatter))

        graph = graphs_operations(graph_A=graph_a, graph_B=graph_b, operation=op, weights=weights) \
            if op else graph_a

        # Save Graphs.
        for graph_type in iter_ui_backend_folders(keep_graph=True):
            save_graph(graph=graph,
                       # We consider the layout for files and keep graphs within the related folder type.
                       out_dir=join(target_dir, graph_type),
                       out_filename=f"{collection_name}",
                       convert_to_radial=True if graph_type == GRAPH_TYPE_RADIAL else False)

        logger.info(f"\n")
        logger.info(f"Dataset is completed and saved in the following locations:")
        for subfolder in iter_ui_backend_folders(keep_desc=True, keep_graph=True):
            logger.info(f"- {os.path.join(target_dir, subfolder, collection_name)}")

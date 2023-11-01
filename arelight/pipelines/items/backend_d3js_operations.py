import logging
import os
from os.path import join

from arekit.common.data.rows_fmt import create_base_column_fmt
from arekit.common.pipeline.context import PipelineContext
from arekit.common.pipeline.items.base import BasePipelineItem

from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web import save_demo_page, iter_ui_backend_folders, GRAPH_TYPE_RADIAL
from arelight.backend.d3js.utils_graph import save_graph


logger = logging.getLogger(__name__)


class D3jsGraphOperationsBackendPipelineItem(BasePipelineItem):

    def __init__(self):
        # Parameters for sampler.
        self.__column_fmts = [create_base_column_fmt(fmt_type="parser")]

    def apply_core(self, input_data, pipeline_ctx):
        assert(isinstance(input_data, PipelineContext))

        graph_a = input_data.provide_or_none("d3js_graph_a")
        graph_b = input_data.provide_or_none("d3js_graph_b")
        op = input_data.provide_or_none("d3js_graph_operations")
        weights = input_data.provide_or_none("d3js_graph_weights")
        target_dir = input_data.provide("d3js_graph_output_dir")
        collection_name = input_data.provide("d3js_collection_name")

        graph = graphs_operations(graph_A=graph_a, graph_B=graph_b, operation=op, weights=weights) \
            if op else graph_a

        # Save Graphs.
        for graph_type in iter_ui_backend_folders(keep_graph=True):
            save_graph(graph=graph,
                       # We consider the layout for files and keep graphs within the related folder type.
                       out_dir=join(target_dir, graph_type),
                       out_filename=f"{collection_name}",
                       convert_to_radial=True if graph_type == GRAPH_TYPE_RADIAL else False)

        # Save Graph description.
        save_demo_page(target_dir=target_dir, collection_name=collection_name)

        print(f"\nDataset is completed and saved in the following locations:")
        for subfolder in iter_ui_backend_folders(keep_desc=True, keep_graph=True):
            print(f"- {os.path.join(target_dir, subfolder, collection_name)}")

        # Launch server to checkout the results (Optionally)
        host_port = input_data.provide_or_none("d3js_host")
        if host_port is not None:
            cmd = f"cd {target_dir} && python -m http.server {host_port}"
            print(f"Launching WEB server for `{target_dir}` dir.")
            logger.info(cmd)
            os.system(cmd)

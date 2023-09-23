import json
import os
import unittest
from os.path import join, dirname, realpath, exists

import pandas as pd

from arelight.backend.d3js.relations_graph_builder import make_graph_from_relations_array
from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web_force import get_force_web_ui
from arelight.backend.d3js.ui_web_radial import get_radial_web_ui
from arelight.backend.d3js.utils_graph import graph_to_radial


class TestBackendD3JS(unittest.TestCase):

    current_dir = dirname(realpath(__file__))
    ORIGIN_DATA_DIR = join(current_dir, "../data")
    TEST_DATA_DIR = join(current_dir, "data")
    TEST_OUT_DIR = join(current_dir, "_out/d3js_test/")

    def compose_radial(self, graph, out_filename):
        data_filepath = join(self.TEST_OUT_DIR, out_filename + ".json")
        with open(data_filepath, "w") as f:
            # Convert to radial graph.
            radial_graph = graph_to_radial(graph)
            content = json.dumps(radial_graph, ensure_ascii=False).encode('utf8').decode()
            f.write(content)

        # Save the result content file.
        # We provide local path, i.e. file in the same folder.
        html_content = get_radial_web_ui(json_data_at_server_filepath=join(out_filename + ".json"))
        with open(join(self.TEST_OUT_DIR, out_filename + ".html"), "w") as f_out:
            f_out.write(html_content)

    def compose_force(self, graph, out_filename):
        data_filepath = join(self.TEST_OUT_DIR, out_filename + ".json")
        with open(data_filepath, "w") as f:
            content = json.dumps(graph, ensure_ascii=False).encode('utf8').decode()
            f.write(content)

        # Save the result content file.
        # We provide local path, i.e. file in the same folder.
        html_content = get_force_web_ui(json_data_at_server_filepath=join(out_filename + ".json"))
        with open(join(self.TEST_OUT_DIR, out_filename + ".html"), "w") as f_out:
            f_out.write(html_content)

    def test(self):

        # Reading source file.
        data = pd.read_csv(join(self.TEST_DATA_DIR, "responses-d3js-backend-sample-data.csv"))

        # Value based visualization.
        relation_type = "WORKS_AS"
        column_name = "relations_pretty_value"

        data_single_type = data[data["relation_type"] == relation_type]
        print(len(data_single_type))

        graph = make_graph_from_relations_array(
            relations=data_single_type[column_name],
            entity_values=data_single_type["entity_values"],
            entity_types=data_single_type["entity_types"],
            min_links=1,
            weights=True
        )

        graph = graphs_operations(
            graph_A=graph, graph_B=graph, operation="SAME",
            min_links=0.01
        )

        if not exists(self.TEST_OUT_DIR):
            os.makedirs(self.TEST_OUT_DIR)

        self.compose_force(graph=graph, out_filename=f"graph_force_{relation_type}")
        self.compose_radial(graph=graph, out_filename=f"graph_radial_{relation_type}")

        # Launch server to checkout the results.
        os.system(f"cd {self.TEST_OUT_DIR} && python -m http.server 8000")

import argparse
import os
from datetime import datetime

from arekit.common.pipeline.base import BasePipelineLauncher

from arelight.backend.d3js.relations_graph_operations import OP_UNION, OP_DIFFERENCE, OP_INTERSECTION
from arelight.backend.d3js.ui_web import GRAPH_TYPE_FORCE
from arelight.backend.d3js.utils_graph import load_graph
from arelight.pipelines.demo.labels.formatter import CustomLabelsFormatter
from arelight.pipelines.demo.result import PipelineResult
from arelight.pipelines.items.backend_d3js_operations import D3jsGraphOperationsBackendPipelineItem
from arelight.run.utils import get_binary_choice, get_list_choice, get_int_choice, is_port_number
from arelight.run.utils_logger import setup_custom_logger


def get_input_with_default(prompt, default_value):
    user_input = input(prompt)
    return user_input.strip() or default_value


def get_graph_path_interactive(text):
    while True:
        folder_path = input(text)

        if not os.path.exists(folder_path):
            print("The specified folder does not exist. Please try again.")
            continue

        force_folder_path = os.path.join(folder_path, GRAPH_TYPE_FORCE)
        if not os.path.exists(force_folder_path):
            print("Looks like it is wrong folder. " 
                  "Please specify folder that is output of ARElight. "
                  f"It should have '{GRAPH_TYPE_FORCE}' subfolder.")
            continue

        json_files = [f for f in os.listdir(force_folder_path) if f.endswith(".json")]
        if not json_files:
            print(f"No JSON files found in the '{GRAPH_TYPE_FORCE}' folder. "
                  "Looks like it is wrong folder. "
                  "Please specify folder that is output of ARElight.")
            continue

        print("Found graphs:")
        for i, json_file in enumerate(json_files, start=1):
            print(f"{i}: {json_file}")

        while True:
            try:
                file_number = int(input("Enter the number of the JSON file you want: "))
                if 1 <= file_number <= len(json_files):
                    selected_file = json_files[file_number - 1]
                    selected_file_path = os.path.join(force_folder_path, selected_file)
                    return selected_file_path

                print("Invalid number. Please enter a valid number.")
            except ValueError:
                print("Invalid input. Please enter a number.")


def create_operations_parser(op_list):

    parser = argparse.ArgumentParser(description="Graph Operations")

    # Providing arguments.
    parser.add_argument("--operation", required=False, choices=op_list,
                        help="Select operation: {ops}".format(ops=",".join(op_list)))
    parser.add_argument("--graph_a_file", required=False,
                        help="Specify path to Graph A (.json file in folder <ARElight_output>/force/)")
    parser.add_argument("--graph_b_file", required=False,
                        help="Specify path to Graph B (.json file in folder <ARElight_output>/force/)")
    parser.add_argument("--weights", required=False, choices=['y', 'n'], help="Use weights? (y/n)")
    parser.add_argument("-o", dest='output_dir', required=False, default="output",
                        help="Specify output directory (you can use directory "
                             "of existing output, it will add files to it)")
    parser.add_argument("--name", required=False, help="Specify name of new graph")
    parser.add_argument("--label-names", dest="d3js_label_names", type=str, default="p:pos,n:neg,u:neu")
    parser.add_argument("--description", required=False, help="Specify description of new graph")
    parser.add_argument('--log-file', dest="log_file", default=None, type=str)
    parser.add_argument("--host", required=False, default=None, help="Server port for launching hosting (optional)")

    return parser


if __name__ == '__main__':

    op_list = [OP_UNION, OP_INTERSECTION, OP_DIFFERENCE]

    # Completing list of arguments.
    parser = create_operations_parser(op_list)

    # Parsing arguments.
    args = parser.parse_args()

    # Setup logger
    logger = setup_custom_logger(name="arelight", filepath=args.log_file)

    operation = args.operation if args.operation else get_list_choice(op_list)
    graph_A_file_path = args.graph_a_file if args.graph_a_file else get_graph_path_interactive(
        "Enter the path to the folder for graph_A: ")
    graph_B_file_path = args.graph_b_file if args.graph_b_file else get_graph_path_interactive(
        "Enter the path to the folder for graph_B: ")
    weights = args.weights.lower() == 'y' if args.weights else get_binary_choice("Use weights? (y/n)\n")
    do_host = args.host if is_port_number(args.host) \
        else get_int_choice(prompt="Server port for launching hosting (optional)\n",
                            filter_func=is_port_number)
    output_dir = args.output_dir if args.output_dir else input(
        "Specify name of output folder (you can use an existing output folder to store new graphs): ")

    default_name = "_".join([
        os.path.basename(graph_A_file_path),
        operation,
        os.path.basename(graph_B_file_path)
    ]).replace(".json", "")

    # Setup collection name.
    collection_name = args.name if args.name \
        else get_input_with_default("Specify name of new graph (enter to skip)\n", default_name)
    if not collection_name.endswith(".json"):
        collection_name += ".json"

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_description = (
        f"Graph A: {graph_A_file_path}\n"
        f"Graph B: {graph_B_file_path}\n"
        f"Operation: {operation}\n"
        f"Use wights: {weights}\n"
        f"Generated on: {current_datetime}"
    )

    description = args.description if args.description else \
        get_input_with_default("Specify description of new graph (enter to skip)\n", default_description)

    labels_fmt = {a: v for a, v in map(lambda item: item.split(":"), args.d3js_label_names.split(','))}

    # Launch application.
    BasePipelineLauncher.run(
        pipeline=[D3jsGraphOperationsBackendPipelineItem()],
        pipeline_ctx=PipelineResult({
            # We provide this settings for inference.
            "labels_formatter": CustomLabelsFormatter(**labels_fmt),
            "d3js_graph_output_dir": output_dir,
            "d3js_collection_description": description,
            "d3js_host": str(8000) if do_host else None,
            "d3js_graph_a": load_graph(graph_A_file_path),
            "d3js_graph_b": load_graph(graph_B_file_path),
            "d3js_graph_operations": operation,
            "d3js_collection_name": collection_name,
            "result": None
    }))

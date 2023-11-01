import argparse
import os
from datetime import datetime
from os.path import join
from pathlib import Path

from arekit.common.utils import create_dir_if_not_exists

from arelight.backend.d3js.relations_graph_operations import graphs_operations
from arelight.backend.d3js.ui_web import iter_ui_backend_folders, GRAPH_TYPE_RADIAL
from arelight.backend.d3js.utils_graph import graph_to_radial, save_json, load_graph


def get_operation():
    while True:
        operation = input("Select operation:\n1: UNION\n2: INTERSECTION\n3: DIFFERENCE\n")
        try:
            operation = int(operation)
            if operation in [1, 2, 3]:
                break
            else:
                print("Invalid choice. Please select operation as number 1, 2, or 3")
        except ValueError:
            print("Invalid input. Please enter a number.")
    return {1: "UNION", 2: "INTERSECTION", 3: "DIFFERENCE"}[operation]


def get_binary_choice(prompt):
    while True:
        choice = input(prompt).lower()
        if choice in ['y', 'n']:
            return choice == 'y'
        else:
            print("Invalid input. Please enter 'y' or 'n'.")


def get_input_with_default(prompt, default_value):
    user_input = input(prompt)
    return user_input.strip() or default_value


def get_graph_path(text):
    while True:
        folder_path = input(text)

        if not os.path.exists(folder_path):
            print("The specified folder does not exist. Please try again.")
        else:
            force_folder_path = os.path.join(folder_path, "force")
            if not os.path.exists(force_folder_path):
                print("Looks like it is wrong folder. Please specify folder that is output of ARElight. "
                      "It should have subfolders 'radial' and 'force'")
            else:
                json_files = [f for f in os.listdir(force_folder_path) if f.endswith(".json")]
                if not json_files:
                    print("No JSON files found in the 'force' folder. "
                          "Looks like it is wrong folder. "
                          "Please specify folder that is output of ARElight.")
                else:
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
                            else:
                                print("Invalid number. Please enter a valid number.")
                        except ValueError:
                            print("Invalid input. Please enter a number.")


if __name__ == '__main__':

    # Providing arguments.
    parser = argparse.ArgumentParser(description="Graph Operations")
    parser.add_argument("--operation", required=False, choices=["UNION", "INTERSECTION", "DIFFERENCE"],
                        help="Select operation: UNION, INTERSECTION, DIFFERENCE")
    parser.add_argument("--graph_a_file", required=False,
                        help="Specify path to Graph A (.json file in folder <ARElight_output>/force/)")
    parser.add_argument("--graph_b_file", required=False,
                        help="Specify path to Graph B (.json file in folder <ARElight_output>/force/)")
    parser.add_argument("--weights", required=False, choices=['y', 'n'], help="Use weights? (y/n)")
    parser.add_argument("--output", required=False, help="Specify output directory (you can use directory "
                                                         "of existing output, it will add files to it)")
    parser.add_argument("--name", required=False, help="Specify name of new graph")
    parser.add_argument("--description", required=False, help="Specify description of new graph")
    parser.add_argument("--host", required=False, choices=["y", "n"], help="Run visualization server? (y/n)")

    # Parsing arguments.
    args = parser.parse_args()

    operation = args.operation if args.operation else get_operation()
    graph_A_file_path = args.graph_a_file if args.graph_a_file else get_graph_path(
        "Enter the path to the folder (ARElight output) for graph_A: ")
    graph_B_file_path = args.graph_b_file if args.graph_b_file else get_graph_path(
        "Enter the path to the folder (ARElight output) for graph_B: ")
    weights = args.weights.lower() == 'y' if args.weights else get_binary_choice("Use weights? (y/n)\n")
    do_host = args.host.lower() in ['y', 'yes'] if args.host else get_binary_choice("Run visualisation server? (y/n)\n")
    output = args.output if args.output else input(
        "Specify name of output folder (you can use an existing output folder to store new graphs): ")

    default_name = "_".join([
        os.path.basename(graph_A_file_path),
        operation,
        os.path.basename(graph_B_file_path)
    ]).replace(".json", "")

    output_name = args.name if args.name \
        else get_input_with_default("Specify name of new graph (enter to skip)\n", default_name)

    if not output_name.endswith(".json"):
        output_name += ".json"

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

    # Perform operation and generate force and radial variants
    new_G_force = graphs_operations(
        graph_A=load_graph(graph_A_file_path),
        graph_B=load_graph(graph_B_file_path),
        operation=operation,
        weights=weights
    )
    new_G_radial = graph_to_radial(new_G_force)

    # Save results
    output_dir = Path(output)
    for subfolder in iter_ui_backend_folders(keep_desc=True, keep_graph=True):
        create_dir_if_not_exists(join(output_dir, subfolder))

    for graph_type in iter_ui_backend_folders(keep_graph=True):
        g = new_G_radial if graph_type == GRAPH_TYPE_RADIAL else new_G_force
        save_json(g, os.path.join(output_dir, graph_type, output_name))

    for subfolder in iter_ui_backend_folders(keep_desc=True):
        save_json({"description": description}, os.path.join(output_dir, subfolder, output_name))

    print(f"\nDataset is completed and saved in the following locations:")
    for subfolder in iter_ui_backend_folders(keep_desc=True, keep_graph=True):
        print(f"- {os.path.join(output_dir, subfolder, output_name)}")

    if do_host:
        pass
        "run visualization server"

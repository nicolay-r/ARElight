import arelight.backend.d3js.relations_graph_operations as rgo
import arelight.backend.d3js.utils_graph as ug
import os
import json
from pathlib import Path
from datetime import datetime
import argparse


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


def select_graph(message, data_files):
    while True:
        note = "\n".join([f"{idx + 1}: {file}" for idx, file in enumerate(data_files)])
        graph = input(f"Select {message}:\n{note}\n")
        try:
            graph_idx = int(graph) - 1
            if 0 <= graph_idx < len(data_files):
                return data_files[graph_idx]
            else:
                print("Invalid choice. Please select a number from the list.")
        except ValueError:
            print("Invalid input. Please enter a number.")


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


def load_graph(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def parse_arguments():
    parser = argparse.ArgumentParser(description="Graph Operations")
    parser.add_argument("--operation", choices=["UNION", "INTERSECTION", "DIFFERENCE"],
                        help="Select operation: UNION, INTERSECTION, DIFFERENCE")
    parser.add_argument("--graph_a", help="Specify file name for Graph A")
    parser.add_argument("--graph_b", help="Specify file name for Graph B")
    parser.add_argument("--weights", choices=['y', 'n'], help="Use weights? (y/n)")
    parser.add_argument("--name", help="Specify name of new graph")
    parser.add_argument("--description", help="Specify description of new graph")
    return parser.parse_args()


if __name__ == '__main__':
    """ (C) Maxim Kolomeets
    """

    args = parse_arguments()

    # 1. Get operation
    operation = args.operation if args.operation else get_operation()

    # 2. Get graphs A and B
    data_dir = Path(__file__).resolve().parent.parent.parent / "output" / "force"
    data_files = os.listdir(data_dir)
    graph_A = args.graph_a if args.graph_a else select_graph("graph A", data_files)
    graph_B = args.graph_b if args.graph_b else select_graph("graph B", data_files)

    # 3. Ask to use weights in computation or not
    weights = args.weights.lower() == 'y' if args.weights else get_binary_choice("Use weights? (y/n)\n")

    # 4. Specify name and description of the new graph
    default_name = f"{graph_A}_{operation}_{graph_B}".replace(".json", "")
    name = args.name if args.name else get_input_with_default("Specify name of new graph (enter to skip)\n",
                                                              default_name)
    if not name.endswith(".json"):
        name += ".json"

    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    default_description = (
        f"Graph A: {graph_A}\n"
        f"Graph B: {graph_B}\n"
        f"Operation: {operation}\n"
        f"Use wights: {weights}\n"
        f"Generated on: {current_datetime}"
    )
    description = args.description if args.description \
        else get_input_with_default("Specify description of new graph (enter to skip)\n",
                                    default_description)

    # 5. Load graphs
    graph_A_data = load_graph(os.path.join(data_dir, graph_A))
    graph_B_data = load_graph(os.path.join(data_dir, graph_B))

    # 6. Perform operation and generate force and radial variants
    new_G_force = rgo.graphs_operations_weighted(
        graph_A=graph_A_data,
        graph_B=graph_B_data,
        operation=operation,
        min_links=0.001,
        weights=weights
    )
    new_G_radial = ug.graph_to_radial(new_G_force)

    # 7. Save results
    output_dir = Path(__file__).resolve().parent.parent.parent / "output"
    save_json(new_G_force, os.path.join(output_dir, "force", name))
    save_json(new_G_radial, os.path.join(output_dir, "radial", name))
    save_json({"description": description}, os.path.join(output_dir, "descriptions", name))

    print(f"\nDataset {name} is completed and saved in the following locations:")
    print(f"- {output_dir / 'force' / name}")
    print(f"- {output_dir / 'radial' / name}")
    print(f"- {output_dir / 'descriptions' / name}\n")

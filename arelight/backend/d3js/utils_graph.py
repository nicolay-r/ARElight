import json
from os import makedirs
from os.path import exists, join


def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)


def load_graph(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


def graph_to_radial(graph):
    """ (C) Maxim Kolomeets
    """
    radial_nodes = {}
    for node in graph["nodes"]:
        radial_node = {"name": node["id"], "w": node["c"], "imports": []}
        radial_nodes[node["id"]] = radial_node
    for link in graph["links"]:
        source_node = link["source"]
        target_node = link["target"]
        radial_nodes[target_node]["imports"].append({
            "name": source_node,
            "w": link["c"],
            "sent": link["sent"]
        })
    return list(radial_nodes.values())


def save_graph(graph, out_dir, out_filename, convert_to_radial=True):

    if not exists(out_dir):
        makedirs(out_dir)

    # Make sure that we have no extention related to the expected format.
    no_ext_basename = out_filename.replace(".json", "")
    target_filepath = join(out_dir, f"{no_ext_basename}.json")
    # Convert to radial graph.
    radial_graph = graph_to_radial(graph) if convert_to_radial else graph
    save_json(data=radial_graph, file_path=target_filepath)

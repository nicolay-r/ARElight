import json
from os import makedirs
from os.path import exists, join


def graph_to_radial(graph):
    """ (C) Maxim Kolomeets
    """
    nodes_ = {}

    for n in graph["nodes"]:
        nodes_[n["id"]] = {"w": n["c"]}

    for l in graph["links"]:

        if l["target"] not in nodes_:
            continue

        target = nodes_[l["target"]]

        if "imports" not in target:
           target["imports"] = []

        target["imports"].append({
            "name": l["source"],
            "w": l["c"],
            "sent": l["sent"]
        })

    nodes = []
    for n_ in nodes_:
        n = nodes_[n_]
        n["name"] = n_
        nodes.append(n)
    return nodes


def save_graph(graph, out_dir, out_filename, convert_to_radial=True):

    if not exists(out_dir):
        makedirs(out_dir)

    data_filepath = join(out_dir, "{}.json".format(out_filename))

    with open(data_filepath, "w") as f:
        # Convert to radial graph.
        radial_graph = graph_to_radial(graph) if convert_to_radial else graph
        json_content = json.dumps(radial_graph, ensure_ascii=False).encode('utf8').decode()
        f.write(json_content)


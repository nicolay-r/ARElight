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
        if "imports" not in nodes_[l["target"]]:
            nodes_[l["target"]]["imports"] = []
        nodes_[l["target"]]["imports"].append({
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


def save_graph(graph, ui_func, out_dir, out_filename, convert_to_radial=True, do_save_template=True):
    assert(callable(ui_func))

    if not exists(out_dir):
        makedirs(out_dir)

    data_filepath = join(out_dir, out_filename + ".json")

    with open(data_filepath, "w") as f:
        # Convert to radial graph.
        radial_graph = graph_to_radial(graph) if convert_to_radial else graph
        json_content = json.dumps(radial_graph, ensure_ascii=False).encode('utf8').decode()
        f.write(json_content)

    # Setup path to JSON content in template.
    if do_save_template:
        json_data_at_server_filepath = join(out_filename + ".json")
    else:
        json_data_at_server_filepath = join(out_dir, out_filename + ".json")

    # Save the result content file.
    # We provide local path, i.e. file in the same folder.
    html_content = ui_func(json_data_at_server_filepath=json_data_at_server_filepath)

    if do_save_template:
        with open(join(out_dir, out_filename + ".html"), "w") as f_out:
            f_out.write(html_content)

    return html_content

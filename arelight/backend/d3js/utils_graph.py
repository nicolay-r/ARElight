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

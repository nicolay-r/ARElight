def graphs_operations(graph_A, graph_B, operation="PLUS", min_links=0.01):
    """ operation: string
            Operation type.
        min_links: float
            we could change this parameter in between [0.001, 0.999]

        (C) Maxim Kolomeets
    """

    links_ = {}

    # convert links of graph A to dict
    links_A = {}
    for link_A in graph_A["links"]:
        l = link_A["source"] + "___" + link_A["target"] + "***" + link_A["sent"]
        if l not in links_A:
            links_A[l] = 0
        links_A[l] += link_A["c"]

    if operation in ["PLUS", "MINUS"]:
        links_ = links_A
        # add or subtract links of graph B
        for link in graph_B["links"]:
            l = link["source"] + "___" + link["target"] + "***" + link["sent"]
            if l not in links_:
                links_[l] = 0
            if operation == "PLUS":
                links_[l] += link["c"]
            else:
                links_[l] -= link["c"]

    if operation in ["SAME", "DIFF"]:
        A_max, B_max = max(links_A.values()), max(map(lambda l: l["c"], graph_B["links"]))
        for l in links_A:
            links_A[l] = links_A[l] / A_max

        if operation == "SAME":
            for link_B in graph_B["links"]:
                l_B = link_B["source"] + "___" + link_B["target"] + "***" + link_B["sent"]
                c = link_B["c"] / B_max
                if l_B in links_A:
                    if c < links_A[l_B]:
                        links_[l_B] = c
                    else:
                        links_[l_B] = links_A[l_B]

        if operation == "DIFF":
            for link_B in graph_B["links"]:
                l_B = link_B["source"] + "___" + link_B["target"] + "***" + link_B["sent"]
                c = link_B["c"] / B_max
                if l_B in links_A and links_A[l_B] - c > 0:
                    links_[l_B] = links_A[l_B] - c

    links = []
    used_nodes = {}
    for s_t in links_:
        if links_[s_t] >= min_links:
            s = s_t.split("___")[0]
            t = s_t.split("___")[1].split("***")[0]
            sent = s_t.split("___")[1].split("***")[1]
            links.append({
                "source": s,
                "target": t,
                "c": links_[s_t],
                "sent": sent})
            if s not in used_nodes:
                used_nodes[s] = 0
            if t not in used_nodes:
                used_nodes[t] = 0
            used_nodes[s] += links_[s_t]
            used_nodes[t] += links_[s_t]
    nodes = []
    for id in used_nodes:
        nodes.append({"id": id, "c": used_nodes[id]})
    return {"nodes": nodes, "links": links}


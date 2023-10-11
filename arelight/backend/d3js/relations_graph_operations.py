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


def graphs_operations_weighted(graph_A, graph_B, operation="UNION", min_links=0.01, weights=True):
    """
    Perform graph operations and return the resulting graph.

    (C) Maxim Kolomeets

    Parameters:
        graph_A, graph_B: dict
            The input graphs to operate on.
        operation: str, optional
            The operation to perform, by default "UNION".
        min_links: float, optional
            A threshold parameter, by default 0.01.
        weights: bool, optional
            Whether to use weights in the computation, by default True.

    Returns:
        dict: The resulting graph after performing the operation.
    """

    print(f"\nPerforming {operation} on graphs...")

    def assign_weights(graph, weight=1):
        """Assign weights to nodes and links in a graph."""
        for element in ["nodes", "links"]:
            for item in graph[element]:
                item["c"] = weight

    def link_key(link):
        """Generate a key for a link."""
        return f"{link['source']}___{link['target']}***{link['sent']}"

    def normalize_links(links, max_val):
        """Normalize link weights by dividing by max_val."""
        return {k: v / max_val for k, v in links.items()}

    # If weights are not used, assign default weights to graphs.
    if not weights:
        assign_weights(graph_A)
        assign_weights(graph_B)

    # Convert links of graph A to a dictionary with link_key as keys.
    links_A = {}
    for link_A in graph_A["links"]:
        l = link_key(link_A)
        links_A[l] = links_A.get(l, 0) + link_A["c"]

    links_ = {}

    # Different operations for UNION and DIFFERENCE
    if operation in ["UNION", "MINUS"]:
        links_ = links_A.copy()
        for link_B in graph_B["links"]:
            l = link_key(link_B)
            links_[l] = links_.get(l, 0) + (link_B["c"] if operation == "PLUS" else -link_B["c"])

    # Different operations for INTERSECTION and DIFFERENCE
    elif operation in ["INTERSECTION", "DIFFERENCE"]:
        A_max, B_max = max(links_A.values()), max(l["c"] for l in graph_B["links"])
        links_A = normalize_links(links_A, A_max)

        for link_B in graph_B["links"]:
            l = link_key(link_B)
            c = link_B["c"] / B_max
            if l in links_A:
                if operation == "INTERSECTION":
                    links_[l] = min(c, links_A[l])
                elif links_A[l] - c > 0:
                    links_[l] = links_A[l] - c

    # Construct the resulting graph.
    links, used_nodes = [], {}
    for s_t, c in links_.items():
        if c >= min_links:
            s, t_sent = s_t.split("___")
            t, sent = t_sent.split("***")
            links.append({"source": s, "target": t, "c": c, "sent": sent})
            used_nodes[s] = used_nodes.get(s, 0) + c
            used_nodes[t] = used_nodes.get(t, 0) + c

    nodes = [{"id": id, "c": c} for id, c in used_nodes.items()]
    result_graph = {"nodes": nodes, "links": links}

    # Assign weights if not used.
    if not weights:
        assign_weights(result_graph)

    return result_graph

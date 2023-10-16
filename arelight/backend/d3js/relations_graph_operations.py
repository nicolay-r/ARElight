def graphs_operations_weighted(graph_A, graph_B, operation="UNION", weights=True):
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
            print(graph)
            for idx, item in enumerate(graph[element]):
                print(item)
                graph[element][idx]["c"] = weight
                #item["c"] = weight

    def link_key(link):
        """Generate a key for a link."""
        return f"{link['source']}___{link['target']}***{link['sent']}"

    def normalize_links(links, max_val):
        """Normalize link weights by dividing by max_val."""
        return {k: v / max_val for k, v in links.items()}

    print("GRAPH A", graph_A)
    print("GRAPH B", graph_B)

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
            links_[l] = links_.get(l, 0) + (link_B["c"] if operation == "UNION" else -link_B["c"])

    # Different operations for INTERSECTION and DIFFERENCE
    elif operation in ["INTERSECTION", "DIFFERENCE"]:
        A_max, B_max = max(l["c"] for l in graph_A["links"]), max(l["c"] for l in graph_B["links"])
        print(A_max)
        print(B_max)
        links_A = normalize_links(links_A, A_max)
        print(links_A)

        for link_B in graph_B["links"]:
            l = link_key(link_B)
            c = link_B["c"] / B_max
            if l in links_A:
                print("HERE2")
                if operation == "INTERSECTION":
                    links_[l] = min(c, links_A[l])
                elif links_A[l] - c > 0:
                    print(links_A[l], c)
                    links_[l] = links_A[l] - c

    # Construct the resulting graph.
    links, used_nodes = [], {}
    for s_t, c in links_.items():
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

    #print("RESULT GRAPH", result_graph)

    return result_graph

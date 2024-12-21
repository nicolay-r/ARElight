import logging
import warnings


OP_UNION = "UNION"
OP_INTERSECTION = "INTERSECTION"
OP_DIFFERENCE = "DIFFERENCE"

OPERATION_MAP = {}
OPERATION_MAP[OP_UNION] = "+"
OPERATION_MAP[OP_INTERSECTION] = "∩"
OPERATION_MAP[OP_DIFFERENCE] = "-"

logger = logging.getLogger(__name__)


def graphs_operations(graph_A, graph_B, operation=OP_UNION, weights=True):
    """
    Perform graph operations and return the resulting graph.

    Parameters:
        graph_A, graph_B: dict
            The input graphs to operate on.
        operation: str, optional
            The operation to perform, by default "UNION".
        weights: bool, optional
            Whether to use weights in the computation, by default True.

    Returns:
        dict: The resulting graph after performing the operation.
    """

    logger.info(f"\nPerforming {operation} on graphs...")

    def link_key(link):
        """Generate a key for a link."""
        return f"{link['source']}___{link['target']}***{link['sent']}"

    # If weights are not used, assign default weights to graphs.
    if not weights:
        for graph in [graph_A, graph_B]:
            for element in ["nodes", "links"]:
                for item in graph[element]:
                    item["c"] = 1

    # Convert links of graph A to a dictionary with link_key as keys.
    links_A = {link_key(link_A): link_A["c"] for link_A in graph_A["links"]}

    # Different operations for UNION, INTERSECTION, and DIFFERENCE
    if operation == OP_UNION:
        links_B = {link_key(link_B): link_B["c"] for link_B in graph_B["links"]}
        links_ = {k: links_A.get(k, 0) + links_B.get(k, 0) for k in set(links_A) | set(links_B)}

    else:
        A_max = max(links_A.values())
        B_max = max(link_B["c"] for link_B in graph_B["links"])

        links_A = {k: v / A_max for k, v in links_A.items()}
        links_B = {link_key(link_B): link_B["c"] / B_max for link_B in graph_B["links"]}

        links_ = {}
        if operation == OP_INTERSECTION:
            for l, c in links_B.items():
                if l in links_A:
                    links_[l] = min(c, links_A[l])

        if operation == OP_DIFFERENCE:
            for l, c in links_A.items():
                if l in links_B and c - links_B[l] > 0:
                    logger.info("     ", l, c, "=>", l, links_B[l])
                    links_[l] = c - links_B[l]
                if l not in links_B:
                    links_[l] = c

    # Normalize link weights after the operation
    try:
        max_c = max(links_.values())
    except ValueError:
        warnings.warn("The result graph is empty.\nThis may be due to absolute no similarity or absolute "
                      "no difference between graph A and B\n(in dependent on which operation you perform)")
        return {"nodes": [{"id": "GPE.EMPTY_GRAPH(no_similarity_OR_no_difference)", "c": 1}],
                "links": []}

    links_ = {k: v / max_c for k, v in links_.items()}

    # Construct the resulting graph.
    links, used_nodes = [], {}
    for s_t, c in links_.items():
        s, t_sent = s_t.split("___")
        t, sent = t_sent.split("***")
        links.append({"source": s, "target": t, "c": c, "sent": sent})
        used_nodes[s] = used_nodes.get(s, 0) + c
        used_nodes[t] = used_nodes.get(t, 0) + c

    nodes = [{"id": id, "c": c} for id, c in used_nodes.items()]
    if operation == OP_DIFFERENCE:
        basis = list(set(graph_A["basis"]).difference(graph_B["basis"]))
    else:
        basis = list(set(graph_A["basis"]).union(graph_B["basis"]))
    equation = "(" + graph_A["equation"] + ")" + OPERATION_MAP[operation] + "(" + graph_B["equation"] + ")"
    result_graph = {"basis": basis, "equation": equation, "nodes": nodes, "links": links}

    # Assign weights if not used.
    if not weights:
        for element in ["nodes", "links"]:
            for item in result_graph[element]:
                item["c"] = 1

    return result_graph

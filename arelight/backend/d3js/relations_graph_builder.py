def make_graph_from_relations_array(relations, entity_values, entity_types, min_links=1, weights=True):
    """ This is a method composes a dictionary from the relations data between entities.
        (C) Maxim Kolomeets
    """

    relations, entity_values, entity_types = list(relations), list(entity_values), list(entity_types)
    nodes_ = {}
    links_ = {}

    entity_values_flat = []
    for e in entity_values:
        entity_values_flat += e.split(",")
    entity_types_flat = []
    for e in entity_types:
        entity_types_flat += e.split(",")
    entity_map = {}
    for idx, value in enumerate(entity_values_flat):
        entity_map[value] = entity_types_flat[idx]

    for idx, relations in enumerate(filter(lambda r: str(r) != "nan", relations)):
        relations = relations.replace(";;", ";").replace(";->", "->").replace("; ", "")
        if ");(" in relations:
            relations = relations.split(");(")
        else:
            relations = [relations]

        for idx, relation in enumerate(relations):
            relation = relation.replace("(", "").replace(")", "")
            relation = relation.split(";")
            sent = relation[1]

            relation[0] = relation[0].split("->")
            source, target = relation[0][0], relation[0][1]
            source, target = source.replace(",", "").replace("'",""), target.replace(",", "").replace("'","")
            try:
                source_type = entity_map[source]
            except KeyError:
                source_type = "UNKNOWN"
            try:
                target_type = entity_map[target]
            except KeyError:
                target_type = "UNKNOWN"
            if source not in nodes_:
                nodes_[source] = 0
            if target not in nodes_:
                nodes_[target] = 0
            nodes_[source] += 1
            nodes_[target] += 1

            s_t = source_type + "." + source + "___" + target_type + "." + target + "***" + sent
            if s_t not in links_:
                links_[s_t] = 0
            links_[s_t] += 1

    node_max = 0
    link_max = 0

    links = []
    used_nodes = set()
    for s_t in links_:
        if links_[s_t] >= min_links:
            links.append({
                "source": s_t.split("___")[0],
                "target": s_t.split("___")[1].split("***")[0],
                "c": links_[s_t] if weights else 1,
                "sent": s_t.split("___")[1].split("***")[1]})
            used_nodes.add(s_t.split("___")[0])
            used_nodes.add(s_t.split("___")[1].split("***")[0])
            if link_max < links_[s_t]:
                link_max = links_[s_t]

    nodes = []
    for id in nodes_:
        if id in used_nodes:
            nodes.append({"id": id, "c": nodes_[id] if weights else 1})
            if node_max < nodes_[id]:
                node_max = nodes_[id]

    return {"nodes": nodes, "links": links}

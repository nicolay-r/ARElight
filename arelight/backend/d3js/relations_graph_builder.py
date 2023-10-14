def make_graph_from_relations_array(relations, entity_values, entity_types, min_links=1,
                                    weights=True, do_auto_cleaning=True):
    """ This is a method composes a dictionary from the relations data between entities.
        (C) Maxim Kolomeets (Originally)

        relations: list
            list of 3-element list: [SubjectValue, ObjectValue, label_str]
        entity_values: list
        entity_types: list
    """

    META_DOT = '.'
    META_UNDERSCORE = '___'
    META_AST = '***'

    char_map = {
        META_DOT: '·',
        META_UNDERSCORE: '---',
        META_AST: '^^^',
    }

    def __clean_meta(entity_value):
        """ We perform this processing to slightly clean the results obtained from NER or other
            prior annotations that were made in the processing pipeline before this backend.
        """
        if not do_auto_cleaning:
            return entity_value

        # Remove last META_DOT Symbol both from
        while entity_value[-1] == META_DOT:
            entity_value = entity_value[:-1]

        return entity_value

    def __mask_meta(s):
        for patter_orig, pattern_to in char_map.items():
            s = s.replace(patter_orig, pattern_to)
        return s

    nodes_ = {}
    links_ = {}

    entity_values_flat = []
    for e in entity_values:
        entity_values_flat += e
    entity_types_flat = []
    for e in entity_types:
        entity_types_flat += e
    entity_map = {}
    for idx, value in enumerate(entity_values_flat):
        entity_map[value] = entity_types_flat[idx]

    for _, relation in enumerate(filter(lambda r: str(r) != "nan", relations)):
        source, target, label = relation

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

        meta_value = [source_type, META_DOT, source, META_UNDERSCORE, target_type, META_DOT, target, META_AST, label]

        # Replacing patterns affect on the syntax of the result string.
        s_t = "".join([__mask_meta(__clean_meta(v)) if v not in char_map.keys() else v for v in meta_value])

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

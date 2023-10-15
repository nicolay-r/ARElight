from collections import Counter


def make_graph_from_relations_array(relations, entity_values, entity_types, min_links, weights=True):
    """ This is a method composes a dictionary from the relations data between entities.
        (C) Maxim Kolomeets (Originally)

        relations: list
            list of 3-element list: [SubjectValue, ObjectValue, label_str]
        entity_values: list
        entity_types: list
    """

    META_DOT = '.'
    char_map = {META_DOT: '·'}

    def __clean(entity_value):
        """ We perform this processing to slightly clean the results obtained from NER or other
            prior annotations that were made in the processing pipeline before this backend.
        """

        # Remove dot from the end.
        while entity_value[-1] == META_DOT:
            entity_value = entity_value[:-1]

        # Perform masking
        for patter_orig, pattern_to in char_map.items():
            entity_value = entity_value.replace(patter_orig, pattern_to)

        return entity_value

    def __get_type(v):
        return entity_map[v] if v in entity_map else "UNKNOWN"

    nodes_ = Counter()
    links_ = Counter()

    entity_values_flat = []
    for e in entity_values:
        entity_values_flat += e
    entity_types_flat = []
    for e in entity_types:
        entity_types_flat += e
    entity_map = {}
    for idx, value in enumerate(entity_values_flat):
        entity_map[value] = entity_types_flat[idx]

    links_meta = {}
    for _, relation in enumerate(filter(lambda r: str(r) != "nan", relations)):
        source, target, label = relation

        nodes_[source] += 1
        nodes_[target] += 1

        complete_source = ''.join([__get_type(source), META_DOT,  __clean(source)])
        complete_target = ''.join([__get_type(target), META_DOT,  __clean(target)])

        # Replacing patterns affect on the syntax of the result string.
        s_t = "".join([complete_source, complete_target, label])

        links_[s_t] += 1
        if s_t not in links_meta:
            links_meta[s_t] = {
                "source": complete_source,
                "target": complete_target,
                "sent": label
            }

    node_max = 0
    link_max = 0

    links = []
    used_nodes = set()
    for s_t in links_.keys():
        if links_[s_t] >= min_links:
            links.append({
                "source": links_meta[s_t]["source"],
                "target": links_meta[s_t]["target"],
                "c": links_[s_t] if weights else 1,
                "sent": links_meta[s_t]["sent"]})
            used_nodes.add(links_meta[s_t]["source"])
            used_nodes.add(links_meta[s_t]["target"])
            if link_max < links_[s_t]:
                link_max = links_[s_t]

    nodes = []
    for id in nodes_.keys():
        complete_id = __get_type(id) + "." + id
        if complete_id in used_nodes:
            nodes.append({"id": complete_id, "c": nodes_[id] if weights else 1})
            if node_max < nodes_[id]:
                node_max = nodes_[id]

    return {"nodes": nodes, "links": links}

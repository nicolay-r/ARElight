def string_terms_to_list(terms):
    r = []
    for t in terms:
        if isinstance(t, str):
            for i in t.split(' '):
                r.append(i)
        else:
            r.append(t)
    return r

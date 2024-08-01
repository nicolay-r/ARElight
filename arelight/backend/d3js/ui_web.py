GRAPH_TYPE_RADIAL = 'radial'
GRAPH_TYPE_FORCE = 'force'


def iter_ui_backend_folders(keep_graph=False, keep_desc=False):
    if keep_graph:
       yield GRAPH_TYPE_RADIAL
       yield GRAPH_TYPE_FORCE
    if keep_desc:
       yield "descriptions"

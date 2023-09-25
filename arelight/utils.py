import importlib


def auto_import(name):
    """ Import from the external python packages.
    """
    def __get_module(comps_list):
        return importlib.import_module(".".join(comps_list))

    components = name.split('.')
    return getattr(__get_module(components[:-1]), components[-1])


class IdAssigner(object):

    def __init__(self):
        self.__id = 0

    def get_id(self):
        curr_id = self.__id
        self.__id += 1
        return curr_id

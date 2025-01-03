class BasePredictWriter(object):

    def __init__(self):
        self._target = None

    def set_target(self, target):
        self._target = target

    def write(self, header, contents_it, total=None):
        raise NotImplementedError()

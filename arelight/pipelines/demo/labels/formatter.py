from arekit.common.labels.base import NoLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel


class CustomLabelsFormatter(StringLabelsFormatter):

    def __init__(self, p="pos", n="neg", u="neu"):

        stol = {}
        if p is not None:
            stol[p] = PositiveLabel
        if n is not None:
            stol[n] = NegativeLabel
        if u is not None:
            stol[u] = NoLabel

        super(CustomLabelsFormatter, self).__init__(stol=stol)

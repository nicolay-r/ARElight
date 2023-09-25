from arekit.common.labels.base import NoLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel


class TrheeLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        super(TrheeLabelsFormatter, self).__init__(
            stol={
                "pos": PositiveLabel,
                "neg": NegativeLabel,
                "neu": NoLabel,
            })

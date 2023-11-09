from arekit.common.labels.base import NoLabel
from arekit.common.labels.str_fmt import StringLabelsFormatter

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel


class ThreeLabelsFormatter(StringLabelsFormatter):

    def __init__(self):
        super(ThreeLabelsFormatter, self).__init__(
            stol={
                "pos": PositiveLabel,
                "neg": NegativeLabel,
                "neu": NoLabel,
            })

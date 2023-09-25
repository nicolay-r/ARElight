from collections import OrderedDict

from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel


class ThreeLabelScaler(BaseLabelScaler):

    def __init__(self):

        uint_labels = [(NoLabel(), 0),
                       (PositiveLabel(), 1),
                       (NegativeLabel(), 2)]

        int_labels = [(NoLabel(), 0),
                      (PositiveLabel(), 1),
                      (NegativeLabel(), -1)]

        super(ThreeLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                               int_dict=OrderedDict(int_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)

from collections import OrderedDict

from arekit.common.labels.base import NoLabel
from arekit.common.labels.scaler.base import BaseLabelScaler

from arelight.pipelines.demo.labels.base import PositiveLabel, NegativeLabel


class CustomLabelScaler(BaseLabelScaler):

    def __init__(self, p=1, n=2, u=0):
        assert((isinstance(p, int) and p >= 0) or p is None)
        assert((isinstance(n, int) and n >= 0) or n is None)
        assert((isinstance(u, int) and u >= 0) or u is None)

        uint_labels = []
        if p is not None:
            uint_labels.append((PositiveLabel(), p))
        if n is not None:
            uint_labels.append((NegativeLabel(), n))
        if u is not None:
            uint_labels.append((NoLabel(), u))

        super(CustomLabelScaler, self).__init__(uint_dict=OrderedDict(uint_labels),
                                                int_dict=OrderedDict(uint_labels))

    def invert_label(self, label):
        int_label = self.label_to_int(label)
        return self.int_to_label(-int_label)

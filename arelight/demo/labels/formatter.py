from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter

from arelight.demo.labels.base import PositiveLabel, NegativeLabel


class ExperimentRuSentiFramesLabelsFormatter(RuSentiFramesLabelsFormatter):

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel

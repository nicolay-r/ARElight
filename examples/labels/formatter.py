from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter
from arekit.contrib.source.rusentrel.labels_fmt import RuSentRelLabelsFormatter

from examples.labels.base import NegativeLabel, PositiveLabel


class RuSentRelExperimentLabelsFormatter(RuSentRelLabelsFormatter):

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel


class ExperimentRuSentiFramesLabelsFormatter(RuSentiFramesLabelsFormatter):

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel
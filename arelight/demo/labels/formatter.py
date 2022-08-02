from arekit.contrib.source.rusentiframes.labels_fmt import RuSentiFramesLabelsFormatter

from arekit.common.labels.base import Label


class PositiveLabel(Label):
    pass


class NegativeLabel(Label):
    pass


class ExperimentRuSentiFramesLabelsFormatter(RuSentiFramesLabelsFormatter):

    @classmethod
    def _positive_label_type(cls):
        return PositiveLabel

    @classmethod
    def _negative_label_type(cls):
        return NegativeLabel

from arekit.common.opinions.collection import OpinionCollection


class CustomOpinionOperations(object):

    def __init__(self, labels_formatter, exp_io, synonyms, neutral_labels_fmt):
        super(CustomOpinionOperations, self).__init__()
        self.__labels_formatter = labels_formatter
        self.__exp_io = exp_io
        self.__synonyms = synonyms
        self.__neutral_labels_fmt = neutral_labels_fmt

    def create_opinion_collection(self, opinions=None):
        return OpinionCollection(opinions=[] if opinions is None else opinions,
                                 synonyms=self.__synonyms,
                                 error_on_duplicates=True,
                                 error_on_synonym_end_missed=True)

from arekit.contrib.experiment_rusentrel.labels.scalers.three import ThreeLabelScaler
from arekit.contrib.experiment_rusentrel.labels.scalers.two import TwoLabelScaler
from arekit.contrib.experiment_rusentrel.synonyms.collection import StemmerBasedSynonymCollection
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups
from arekit.processing.lemmatization.mystem import MystemWrapper


def create_labels_scaler(labels_count):
    assert (isinstance(labels_count, int))

    if labels_count == 2:
        return TwoLabelScaler()
    if labels_count == 3:
        return ThreeLabelScaler()

    raise NotImplementedError("Not supported")


def read_synonyms_collection(filepath):

    def __iter_groups(filepath):
        with open(filepath, 'r') as file:
            for group in iter_synonym_groups(file):
                yield group

    return StemmerBasedSynonymCollection(
        iter_group_values_lists=__iter_groups(filepath),
        stemmer=MystemWrapper(),
        is_read_only=False,
        debug=False)

from arekit.common.text.stemmer import Stemmer
from arekit.contrib.experiment_rusentrel.synonyms.collection import StemmerBasedSynonymCollection
from arekit.contrib.source.rusentrel.utils import iter_synonym_groups


def iter_groups(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        for group in iter_synonym_groups(file):
            yield group


def read_synonyms_collection(synonyms_filepath, stemmer):
    assert(iter_synonym_groups(stemmer, Stemmer))

    synonyms = StemmerBasedSynonymCollection(
        iter_group_values_lists=iter_groups(synonyms_filepath),
        stemmer=stemmer,
        is_read_only=False, debug=False)

    return synonyms

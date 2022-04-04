from itertools import chain

from arekit.common.experiment.data_type import DataType
from arekit.common.folding.nofold import NoFolding
from arekit.contrib.experiment_rusentrel.exp_ds.utils import read_ruattitudes_in_memory
from arekit.contrib.source.rusentrel.io_utils import RuSentRelIOUtils

from network.nn.embedding import RusvectoresEmbedding


class Common:

    @staticmethod
    def ra_doc_id_func(doc_id):
        return 10000 + doc_id

    @staticmethod
    def create_folding(rusentrel_version, ruattitudes_version, doc_id_func):
        rsr_indices = list(RuSentRelIOUtils.iter_collection_indices(rusentrel_version))

        ra_indices_dict = dict()
        if ruattitudes_version is not None:
            ra_indices_dict = read_ruattitudes_in_memory(version=ruattitudes_version,
                                                         keep_doc_ids_only=True,
                                                         doc_id_func=doc_id_func)

        return NoFolding(doc_ids_to_fold=list(chain(rsr_indices, ra_indices_dict.keys())),
                         supported_data_types=[DataType.Train])

    @staticmethod
    def create_exp_name(rusentrel_version, ra_version, folding_type):
        return "".join(["rsr-{v}".format(v=rusentrel_version.value),
                        "-ra-{v}".format(v=ra_version.value) if ra_version is not None else "",
                        "-{ft}".format(ft=folding_type.value)])

    @staticmethod
    def create_exp_name_suffix(use_balancing, terms_per_context, dist_in_terms_between_att_ends):
        """ Provides an external parameters that assumes to be synchronized both
            by serialization and training experiment stages.
        """
        assert(isinstance(use_balancing, bool))
        assert(isinstance(terms_per_context, int))
        assert(isinstance(dist_in_terms_between_att_ends, int) or dist_in_terms_between_att_ends is None)

        # You may provide your own parameters out there
        params = [
            u"balanced" if use_balancing else u"nobalance",
            u"tpc{}".format(terms_per_context)
        ]

        if dist_in_terms_between_att_ends is not None:
            params.append(u"dbe{}".format(dist_in_terms_between_att_ends))

        return u'-'.join(params)

    @staticmethod
    def load_rusvectores_embedding(filepath, stemmer):
        embedding = RusvectoresEmbedding.from_word2vec_format(filepath=filepath, binary=True)
        embedding.set_stemmer(stemmer)
        return embedding


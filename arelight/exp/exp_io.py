import os
from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils


class InferIOUtils(RuSentRelExperimentNetworkIOUtils):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(InferIOUtils, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def __create_target(self, doc_id, data_type):
        filename = "result_d{doc_id}_{data_type}.txt".format(doc_id=doc_id, data_type=data_type.name)
        return os.path.join(self._get_target_dir(), filename)

    def _get_experiment_sources_dir(self):
        return self.__output_dir

    def create_opinion_collection_target(self, doc_id, data_type, check_existance=False):
        return self.__create_target(doc_id=doc_id, data_type=data_type)

    def create_result_opinion_collection_target(self, doc_id, data_type, epoch_index):
        return self.__create_target(doc_id=doc_id, data_type=data_type)

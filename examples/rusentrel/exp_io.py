from arekit.contrib.experiment_rusentrel.model_io.tf_networks import RuSentRelExperimentNetworkIOUtils
from examples.args.const import OUTPUT_DIR


class CustomRuSentRelNetworkExperimentIO(RuSentRelExperimentNetworkIOUtils):

    def try_prepare(self):
        pass

    def _get_experiment_sources_dir(self):
        return OUTPUT_DIR

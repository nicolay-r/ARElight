import importlib
from arelight.ner.base import BaseNER


class DeepPavlovNER(BaseNER):

    def __init__(self, model_name):

        # Dynamic libraries import.
        deeppavlov = importlib.import_module("deeppavlov")
        build_model = deeppavlov.build_model
        self.__ner_model = build_model(model_name, download=True, install=True)

    # region Properties

    def _forward(self, sequences):
        return self.__ner_model(sequences)
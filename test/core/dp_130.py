# This is an adapter implementation for dp_130
# deeppavlov==1.3.0

from deeppavlov import build_model
from bulk_ner.src.ner.base import BaseNER


class DeepPavlovNER(BaseNER):

    def __init__(self, model, download=True, install=True):
        self.__ner_model = build_model(model, download=download, install=install)

    def _forward(self, sequences):
        """ This function is expected to return list of terms
            alongside with the list of labels in CONLL format.
        """
        return self.__ner_model(sequences)
import importlib

from arelight.ner.base import BaseNER


class DeepPavlovNER(BaseNER):

    DEFAULT_MODEL = "ontonotes_mult"

    def __init__(self, model_cfg=None):
        assert(isinstance(model_cfg, str) or model_cfg is None)

        # Dynamic libraries import.
        deeppavlov = importlib.import_module("deeppavlov")
        build_model = deeppavlov.build_model
        configs = deeppavlov.configs

        model_cfg = DeepPavlovNER.DEFAULT_MODEL if model_cfg is None else model_cfg

        # Mapping list of the available models.
        __models = {
            DeepPavlovNER.DEFAULT_MODEL: configs.ner.ner_ontonotes_bert_mult,
            "ontonotes_eng": configs.ner.ner_ontonotes_bert,
        }

        self.__ner_model = build_model(__models[model_cfg],
                                       download=True)

    # region Properties

    def _extract_tags(self, sequences):
        tokens, labels = self.__ner_model(sequences)
        gathered_labels_seq = []
        for i, sequence in enumerate(sequences):
            _, labels = self.__tokens_to_terms(terms=sequence, tokens=tokens[i], labels=labels[i])
            gathered_labels_seq.append(self.__gather(labels))
        return gathered_labels_seq

    @staticmethod
    def __tokens_to_terms(terms, tokens, labels):
        def __cur_term():
            return len(joined_tokens) - 1

        assert (len(labels) == len(tokens))

        terms_lengths = [len(term) for term in terms]
        current_lengths = [0] * len(terms)
        joined_tokens = [[]]
        joined_labels = [[]]
        for i, token in enumerate(tokens):
            if current_lengths[__cur_term()] == terms_lengths[__cur_term()]:
                joined_tokens.append([])
                joined_labels.append([])
            joined_tokens[-1].append(token)
            joined_labels[-1].append(labels[i])
            current_lengths[__cur_term()] += len(token)

        return joined_tokens, joined_labels

    @staticmethod
    def __gather(labels_in_lists):
        return [labels[0] if len(labels) == 1 else DeepPavlovNER.__gather_many(labels)
                for labels in labels_in_lists]

    @staticmethod
    def __gather_many(labels):
        return 'O'

from arekit.common.entities.str_fmt import StringEntitiesFormatter
from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext


class BertSerializationContext(ExperimentSerializationContext):

    def __init__(self, label_scaler, terms_per_context, str_entity_formatter,
                 annotator, name_provider, data_folding):
        assert(isinstance(str_entity_formatter, StringEntitiesFormatter))
        assert(isinstance(terms_per_context, int))

        super(BertSerializationContext, self).__init__(annot=annotator,
                                                       name_provider=name_provider,
                                                       label_scaler=label_scaler,
                                                       data_folding=data_folding)

        self.__terms_per_context = terms_per_context
        self.__str_entity_formatter = str_entity_formatter

    @property
    def StringEntityFormatter(self):
        return self.__str_entity_formatter

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

from arekit.common.experiment.api.ctx_serialization import ExperimentSerializationContext


class BertSerializationContext(ExperimentSerializationContext):

    def __init__(self, label_scaler, terms_per_context, name_provider):
        assert(isinstance(terms_per_context, int))
        super(BertSerializationContext, self).__init__(name_provider=name_provider, label_scaler=label_scaler)
        self.__terms_per_context = terms_per_context

    @property
    def TermsPerContext(self):
        return self.__terms_per_context

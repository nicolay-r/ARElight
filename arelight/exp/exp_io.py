class InferIOUtils(object):

    def __init__(self, output_dir, exp_ctx):
        assert(isinstance(output_dir, str))
        super(InferIOUtils, self).__init__(exp_ctx=exp_ctx)
        self.__output_dir = output_dir

    def _get_experiment_sources_dir(self):
        return self.__output_dir

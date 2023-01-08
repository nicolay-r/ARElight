import gzip

from arekit.common.utils import progress_bar_iter, create_dir_if_not_exists

from arelight.predict_writer import BasePredictWriter


class TsvPredictWriter(BasePredictWriter):

    def __init__(self):
        super(TsvPredictWriter, self).__init__()
        self.__col_separator = '\t'
        self.__f = None

    def __write(self, params):
        line = "{}\n".format(self.__col_separator.join([str(p) for p in params]))
        self.__f.write(line.encode())

    def write(self, title, contents_it):
        self.__write(title)

        wrapped_it = progress_bar_iter(iterable=contents_it,
                                       desc='Writing output',
                                       unit='rows')

        for contents in wrapped_it:
            self.__write(contents)

    # region base

    def __enter__(self):
        create_dir_if_not_exists(self._target)
        self.__f = gzip.open(self._target, 'wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__f.close()

    # endregion

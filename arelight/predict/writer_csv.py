import gzip
import logging

from arekit.common.utils import progress_bar_defined, create_dir_if_not_exists

from arelight.predict.writer import BasePredictWriter

logger = logging.getLogger(__name__)


class TsvPredictWriter(BasePredictWriter):

    def __init__(self, log_out=None):
        super(TsvPredictWriter, self).__init__()
        self.__col_separator = '\t'
        self.__f = None
        self.__log_out = log_out

    def __write(self, params):
        line = "{}\n".format(self.__col_separator.join([str(p) for p in params]))
        self.__f.write(line.encode())

    def write(self, header, contents_it, total=None):
        self.__write(header)

        wrapped_it = progress_bar_defined(iterable=contents_it, desc='Writing output (tsv)', unit='rows',
                                          total=total, file=self.__log_out)

        for contents in wrapped_it:
            self.__write(contents)

    # region base

    def __enter__(self):
        create_dir_if_not_exists(self._target)
        self.__f = gzip.open(self._target, 'wb')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        logger.info(f"Saved: {self._target}")
        self.__f.close()

    # endregion

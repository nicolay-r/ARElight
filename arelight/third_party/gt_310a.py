# This implementation has been tested for
# googletrans==3.1.0a0


import time

from googletrans import Translator


class GoogleTranslateModel(object):

    def __init__(self, **kwargs):
        self._instance = Translator()

    @staticmethod
    def translate_value(translator, value, src, dest, sec_delay=1, attempts=10):

        import logging
        logger = logging.getLogger()  # get the default logger
        logger.setLevel(50)

        for i in range(attempts):
            try:
                translated = translator.translate(value, dest=dest, src=src)
                return translated.text
            except:
                logger.info("Unable to perform translation. Try {} out of {}.".format(i, attempts))
                time.sleep(sec_delay)

        raise Exception("Can't translate")

    def get_func(self, src, dest, **kwargs):
        # We do auto-import so we not depend on the actually installed library.
        # Translation of the list of data.
        # Returns the list of strings.
        return lambda str_list: [
            GoogleTranslateModel.translate_value(translator=self._instance, value=s, dest=dest, src=src)
            for s in str_list]
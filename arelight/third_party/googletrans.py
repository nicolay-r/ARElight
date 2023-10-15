import time

from googletrans import Translator


class SingletonTranslator(object):
    _instance = None

    def __init__(self):
        raise RuntimeError('Call instance() instead')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = Translator()
        return cls._instance


def translate_value(value, src, dest, attempts=10):
    """ This is a main wrapping for GoogleTranslation API calls.
    """
    translator = SingletonTranslator.instance()

    import logging
    logger = logging.getLogger()  # get the default logger
    logger.setLevel(50)

    for i in range(attempts):
        try:
            translated = translator.translate(value, dest=dest, src=src)
            return translated.text
        except:
            logger.info("Unable to perform translation. Try {} out of {}.".format(i, attempts))
            time.sleep(1)

    raise Exception("Can't translate")

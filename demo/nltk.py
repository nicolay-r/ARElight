import nltk


def import_tokenizer(name, resource_name):
    try:
        nltk.data.find(name)
    except LookupError:
        nltk.download(resource_name)
    return nltk.data.load(name)


def tokenize(language):
    tokenizer = import_tokenizer(name="tokenizers/punkt/{0}.pickle".format(language), resource_name="punkt")
    return tokenizer.tokenize

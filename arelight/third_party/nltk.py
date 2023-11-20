import nltk


def import_tokenizer(name, resource_name):
    try:
        nltk.data.find(name)
    except LookupError:
        nltk.download(resource_name)
    return nltk.data.load(name)

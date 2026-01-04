from io import TextIOWrapper
from zipfile import ZipFile

from arelight.synonyms import iter_synonym_groups
from arelight.utils import auto_import, iter_csv_lines


NER_TYPES = ["ORG", "PERSON", "LOC", "GPE"]


def create_sentence_parser(framework, language):
    if framework == "linesplit":
        return lambda text: [t.strip() for t in text.split('\n')]
    elif framework == "nltk":
        # Using nltk library.
        tokenizer_func = auto_import("arelight.third_party.nltk.import_tokenizer")
        tokenizer = tokenizer_func(name=f'tokenizers/punkt/{language}.pickle', resource_name="punkt")
        return tokenizer.tokenize
    else:
        raise Exception("Framework `{}` is not supported".format(framework))


def iter_content(filepath, csv_column, csv_delimiter, open_func=None):

    open_funcs = {
        '.csv': lambda fp: open(fp, mode="r", encoding="utf-8-sig"),
        '.zip': lambda fp: ZipFile(fp, mode='r'),
        '.txt': lambda fp: open(fp, mode='r')
    }

    if filepath.endswith(".csv"):
        open_func = open_funcs[".csv"] if open_func is None else open_func
        for line in iter_csv_lines(open_func(filepath), column_name=csv_column, delimiter=csv_delimiter):
            yield line
    elif filepath.endswith('.zip'):
        open_func = open_funcs[".zip"] if open_func is None else open_func
        with open_func(filepath) as zip_file:
            for file_name in zip_file.namelist():
                content_it = iter_content(filepath=file_name, csv_column=csv_column, csv_delimiter=csv_delimiter,
                                          open_func=lambda fp: TextIOWrapper(zip_file.open(fp), 'utf-8'))
                for content in content_it:
                    yield content
    else:
        open_func = open_funcs[".txt"] if open_func is None else open_func
        with open_func(filepath) as f:
            yield f.read().rstrip()


def iter_group_values(filepath):

    if filepath is None:
        return None

    with open(filepath, 'r') as file:
        for group in iter_synonym_groups(file):
            yield group


def merge_dictionaries(dict_iter):
    merged_dict = {}
    for d in dict_iter:
        for key, value in d.items():
            if key in merged_dict:
                raise Exception("Key `{}` is already registred!".format(key))
            merged_dict[key] = value
    return merged_dict
import collections
import csv
import importlib
import os
import sys

import requests
from bulk_chain.api import iter_content
from bulk_chain.core.utils import dynamic_init
from tqdm import tqdm


def auto_import(name):
    """ Import from the external python packages.
    """
    def __get_module(comps_list):
        return importlib.import_module(".".join(comps_list))

    components = name.split('.')
    return getattr(__get_module(components[:-1]), components[-1])


def flatten(xss):
    l = []
    for xs in xss:
        if isinstance(xs, collections.abc.Iterable):
            for x in xs:
                l.append(x)
        else:
            l.append(xs)
    return l


def get_default_download_dir():
    """ Refered to NLTK toolkit approach
        https://github.com/nltk/nltk/blob/8e771679cee1b4a9540633cc3ea17f4421ffd6c0/nltk/downloader.py#L1051
    """

    # On Windows, use %APPDATA%
    if sys.platform == "win32" and "APPDATA" in os.environ:
        homedir = os.environ["APPDATA"]

    # Otherwise, install in the user's home directory.
    else:
        homedir = os.path.expanduser("~/")
        if homedir == "~/":
            raise ValueError("Could not find a default download directory")

    return os.path.join(homedir, ".arelight")


def iter_csv_lines(csv_file, column_name, delimiter=","):

    with csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=delimiter)

        if column_name not in csv_reader.fieldnames:
            print(f"Error: {column_name} column not found.")

        for row in csv_reader:
            yield row[column_name]


def download(dest_file_path, source_url, logger):
    """ Refered to https://github.com/nicolay-r/ner-bilstm-crf-tensorflow/blob/master/ner/utils.py
        Simple http file downloader
    """
    if logger is not None:
        logger.info(('Downloading from {src} to {dest}'.format(src=source_url, dest=dest_file_path)))

    sys.stdout.flush()
    datapath = os.path.dirname(dest_file_path)

    if not os.path.exists(datapath):
        os.makedirs(datapath, mode=0o755)

    dest_file_path = os.path.abspath(dest_file_path)

    r = requests.get(source_url, stream=True)
    total_length = int(r.headers.get('content-length', 0))

    with open(dest_file_path, 'wb') as f:
        pbar = tqdm(total=total_length, unit='B', unit_scale=True)
        for chunk in r.iter_content(chunk_size=32 * 1024):
            if chunk:  # filter out keep-alive new chunks
                pbar.update(len(chunk))
                f.write(chunk)


def init_llm(class_filepath, api_key, model_name="meta/meta-llama-3-70b-instruct"):
    return dynamic_init(class_filepath)(
        api_token=api_key,
        model_name=model_name)


def infer_async_batch_it(llm, schema, input_dicts_it, batch_size=1):
    c = iter_content(schema=schema, llm=llm, infer_mode="batch_async", input_dicts_it=input_dicts_it, batch_size=batch_size)
    for batch in c:
        for record in batch:
            print(record)
            yield record


def batch_iter(arr, block_size):
    if block_size <= 0:
        raise ValueError("block_size must be positive")

    for i in range(0, len(arr), block_size):
        yield arr[i:i + block_size]
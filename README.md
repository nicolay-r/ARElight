# ARElight 0.26.2

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.25.2-orange.svg)
[![](https://img.shields.io/badge/demo-0.24.0-purple.svg)](https://guardeec.github.io/arelight_demo/template.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ARElight/blob/v0.24.0/ARElight.ipynb)
[![PyPI downloads](https://img.shields.io/pypi/dm/arelight.svg)](https://pypistats.org/packages/arelight)

<p align="center">
    <img src="logo.png"/>
</p>
<p align="center">
  <a href="https://guardeec.github.io/arelight_demo/template.html">👉<b>DEMO</b>👈</a>
  <br>
  <a href="https://github.com/nicolay-r/ARElight-server">👉GUI server setup👈</a>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities 
in texts. 
This repository is a part of the **ECIR-2024** demo paper: 
[ARElight: Context Sampling of Large Texts for Deep Learning Relation Extraction](https://link.springer.com/chapter/10.1007/978-3-031-56069-9_23).


# Installation

```bash
pip install git+https://github.com/nicolay-r/arelight@v0.26.2
```

Install the demo packages and download the providers:

* [dp_130.py](https://github.com/nicolay-r/nlp-thirdgate/blob/master/ner/dp_130.py)
* [replicate_104.py](https://github.com/nicolay-r/nlp-thirdgate/blob/master/llm/replicate_104.py)
* [googletrans_402.py](https://github.com/nicolay-r/nlp-thirdgate/blob/master/text-translation/googletrans_402.py)

```bash
pip install -r demo/dependencies.txt
sh demo/download_providers.sh
```

# 🚀 API

Pass in-memory rows to `create_inference_pipeline`. Each row has an `id` and a `text`.

```python
from dotenv import load_dotenv
from arekit.common.pipeline.base import BasePipelineLauncher

from arelight.api import create_inference_pipeline
from arelight.predict.provider import IntPredictProvider
from arelight.pipelines.result import PipelineResult
from arelight.run.infer import create_sentence_parser
from arelight.run.utils import merge_dictionaries
from arelight.utils import get_event_loop

from bulk_chain.core.utils import dynamic_init

load_dotenv()  # API_TOKEN in .env is read by the inference pipeline

event_loop = get_event_loop()

pipeline, settings = create_inference_pipeline(
    rows_iter=[
        {"id": "request:0", "text": "Barack Obama met Angela Merkel in Berlin."}
        # ... iterator of texts goes here
    ],
    sampling_args={
        "terms_per_context": 50,
        "docs_limit": 1,
    },
    sentence_parser=create_sentence_parser(framework="nltk", language="english"),
    predict_table_name="bulk_chain",
    collection_target_func=lambda data_type: f"./output/test_samples-{data_type.name.lower()}",
    # Translate the document text. These kwargs are forwarded to the translator.
    translator_args={
        "model": dynamic_init("demo/googletrans_402.py")(),
        "src": "auto",
        "dest": "en",
    },
    # Find named entities to relate.
    ner_args={
        "model": dynamic_init("demo/dp_130.py")(model="ner_ontonotes_bert_mult"),
        "obj_filter": lambda s_obj: s_obj.ObjectType in ["ORG", "PERSON", "LOC", "GPE"],
        "chunk_limit": 128,
    },
    # Classify the attitude between those entities.
    inference_args={
        "model_name": "meta/meta-llama-3-70b-instruct",
        "class_name": "demo/replicate_104.py",
        "task_kwargs": {
            "schema": [{
                "prompt": "Given text: {text_a}TASK: Classify sentiment attitude of [SUBJECT] to [OBJECT]. Reply with a single integer: 0 for neutral, 1 for positive, 2 for negative.",
                "out": "response",
            }],
            "labels_fmt": "u:0,p:1,n:2",
        },
    writer_args={
        "writer": "sqlite3",
        "provider": IntPredictProvider,
    },
    batch_size=1,
    event_loop=event_loop,
)

# Launch Pipeline
BasePipelineLauncher.run(
    pipeline=pipeline,
    pipeline_ctx=PipelineResult(merge_dictionaries(settings)),
    has_input=False)
```

# CLI

The command-line entry point reads files and uses the same providers.

Infer sentiment attitudes from a text file **in English**:
```bash
python3 -m arelight.run.infer \
    --batch-size 10 \
    --from-files "<YOUR-FILES-GOES-HERE>" \
    --sampling-framework "arekit" \
    --sentence-parser "nltk:english" \
    --terms-per-context 50 \
    --translate-provider "demo/googletrans_402.py" \
    --translate-text "auto:en" \
    --ner-provider "demo/dp_130.py" \
    --ner-model-name "ner_ontonotes_bert_mult" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --inference-filename "demo/replicate_104.py" \
    --inference-model-name "meta/meta-llama-3-70b-instruct" \
    --inference-writer "sqlite3" \
    --log-file "arelight.log.txt" \
    -o "output" 
```

> **NOTE:** [Applying ARElight for **non-english texts**](https://github.com/nicolay-r/ARElight/wiki/Language-Specific-Application)
>

<details>
<summary>

### Parameters
</summary>

The complete documentation is available via `-h` flag:
```bash
python3 -m arelight.run.infer -h
```

Parameters:
* `sampling-framework` we consider only `arekit` framework by default.
    * `from-files` -- list of filepaths to the related documents.
      * for the `.csv` files we consider that each line of the particular `column` as a separated document.
        * `csv-sep` -- separator between columns.
        * `csv-column` -- name of the column in CSV file.
    * `collection-name` -- name of the result files based on sampled documents.
    * `terms-per-context` -- total amount of words for a single sample.
    * `sentence-parser` -- parser utilized for document split into sentences. `nltk` is provided by [demo/nltk.py](demo/nltk.py) and passed into the pipeline as a callable.
    * `synonyms-filepath` -- text file with listed synonymous entries, grouped by lines. [[example]](https://github.com/nicolay-r/RuSentRel/blob/master/synonyms.txt).
    * `stemmer` -- for words lemmatization (optional); we support [[PyMystem]](https://pypi.org/project/pymystem3/).
    * NER parameters:  
      * `ner-framework` -- type of the framework:
        * `deeppavlov` -- [[DeepPavlov]](https://docs.deeppavlov.ai/en/master/features/models/NER.html#6.-Models-list) list of models.
        * `transformers` -- [[Transformers]](https://huggingface.co/models?library=transformers&other=named-entity-recognition&sort=downloads) list of models.
      * `ner-model-name` -- model name within utilized NER framework.
      * `ner-types` -- list of types to be considered for annotation, separated by `|`.
    * `docs-limit` -- the total limit of documents for sampling.
  * [Translation specific parameters](https://github.com/nicolay-r/ARElight/wiki/Language-Specific-Application#any-other-languages)
      * `translate-framework` -- text translation backend (optional); we support [[googletrans]](https://github.com/nicolay-r/ARElight/blob/a17088a98729e3092de1666bef9ba8327ef30b80/arelight/run/utils.py#L31)
      * `translate-entity` -- (optional) source and target language supported by backend, separated by `:`.
      * `translate-text` -- (optional) source and target language supported by backend, separated by `:`.
* `bulk_chain` -- LLM framework, utilized for relations classification) [[bulk-chain]](https://github.com/nicolay-r/bulk-chain)
    * `API_TOKEN` in `.env` -- API key, loaded with `python-dotenv` inside the inference pipeline. It is not an `inference_args` field.
    * `inference-filename` -- Inference implemenation filename; passed through as `class_name`. We use `replicate_104.py` from [nlp-thirdgate](https://github.com/nicolay-r/nlp-thirdgate/blob/master/llm/replicate_104.py) by default
    * `inference-model-name` -- Name of the model, passed through as `model_name`. For the Replicate provider we use: `meta/meta-llama-3-70b-instruct`
    * `inference-writer` -- How to save the output results; by default we use `sqlite3` for storing classified results.
* `labels-fmt` -- list of the mappings from `label` to integer value; is a `p:1,n:2,u:0` by default, where:
    * `p` -- positive label, which is mapped to `1`.
    * `n` -- negative label, which is mapped to `2`.
    * `u` -- undefined label (optional), which is mapped to `0`.
* `-o` -- output folder for result collections and demo.

Framework parameters mentioned above as well as their related setups might be omitted.

</details>

The GUI server for this pipeline lives in [ARElight-server](https://github.com/nicolay-r/ARElight-server).

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>

## How to cite
Our one and my personal interest is to help you better explore and analyze attitude and relation extraction related tasks with ARElight. 
A great research is also accompanied with the faithful reference. 
if you use or extend our work, please cite as follows:

```bibtex
@inproceedings{rusnachenko2024arelight,
  title={ARElight: Context Sampling of Large Texts for Deep Learning Relation Extraction},
  author={Rusnachenko, Nicolay and Liang, Huizhi and Kolomeets, Maxim and Shi, Lei},
  booktitle={European Conference on Information Retrieval},
  year={2024},
  organization={Springer}
}
```

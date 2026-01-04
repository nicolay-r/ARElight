# ARElight 0.26.0

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
pip install git+https://github.com/nicolay-r/arelight@v0.26.0
```

# GUI Interface 

Since the version `0.25.0` ARElight has an updated GUI server
![image](https://github.com/user-attachments/assets/b7a1189a-b5b0-479f-8413-f8a16801e06b)
<p align="center">
  <a href="https://github.com/nicolay-r/ARElight-server">👉GUI server setup👈</a>
</p>

## Usage: Inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ARElight/blob/v0.24.0/ARElight.ipynb)

**UPDATE for LLM usage**: We support [Replicate](https://replicate.com/) for Inference.

> **NOTE:** You have to **download providers** to start with usage script below:
> 1. https://github.com/nicolay-r/nlp-thirdgate/blob/master/ner/dp_130.py
> 2. https://github.com/nicolay-r/nlp-thirdgate/blob/master/llm/replicate_104.py
> 3. https://github.com/nicolay-r/nlp-thirdgate/blob/master/text-translation/googletrans_402.py

Infer sentiment attitudes from text file **in English**:
```bash
python3 -m arelight.run.infer \
    --batch-size 10 \
    --from-files "<YOUR-FILES-GOES-HERE>" \
    --sampling-framework "arekit" \
    --sentence-parser "nltk:english" \
    --terms-per-context 50 \
	--translate-provider "gt_402.py" \
	--translate-text "auto:en" \
    --ner-provider "dp_130.py" \
    --ner-model-name "ner_ontonotes_bert_mult" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --inference-api "<YOUR-API-GOES-HERE>" \
    --inference-filename "replicate_104.py" \
    --inference-model-name "meta/meta-llama-3-70b-instruct" \
    --inference-writer "sqlite3" \
    --backend "d3js_graphs" \
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
    * `sentence-parser` -- parser utilized for document split into sentences; list of the [[supported parsers]](https://github.com/nicolay-r/ARElight/blob/a17088a98729e3092de1666bef9ba8327ef30b80/arelight/run/utils.py#L15).
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
    * `inference-api` -- API key 
    * `inference-filename` -- Inference implemenation filename; we use `replicate_104.py` from [nlp-thirdgate](https://github.com/nicolay-r/nlp-thirdgate/blob/master/llm/replicate_104.py) by default
    * `inference-model-name` -- Name of the model; for the Replicate provider we use: `meta/meta-llama-3-70b-instruct`
    * `inference-writer` -- How to save the output results; by default we use `sqlite3` for storing classified results.
* `labels-fmt` -- list of the mappings from `label` to integer value; is a `p:1,n:2,u:0` by default, where:
    * `p` -- positive label, which is mapped to `1`.
    * `n` -- negative label, which is mapped to `2`.
    * `u` -- undefined label (optional), which is mapped to `0`.
* `backend` -- type of the backend (`d3js_graphs` by default).
    * `host` -- port on which we expect to launch localhost server.
    * `label-names` -- default mapping is `p:pos,n:neg,u:neu`.
* `-o` -- output folder for result collections and demo.

Framework parameters mentioned above as well as their related setups might be omitted.

</details>
 

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

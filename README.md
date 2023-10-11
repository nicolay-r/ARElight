# ARElight 0.24.0

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.24.0-orange.svg)

### :point_right: [DEMO]() :point_left:

<p align="center">
    <img src="logo.png"/>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities 
in texts.


# Installation

```bash
pip install git+https://github.com/nicolay-r/arelight@v0.24.0
```

## Usage

Infer sentiment attitudes from text file:
```bash
python3 -m arelight.run.infer  \
    --sampling-framework "arekit" \
    --ner-model-name "ner_ontonotes_bert_mult" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --terms-per-context 50 \
    --sentence-parser "ru" \
    --tokens-per-context 128 \
    --bert-framework "opennre" \
    --batch-size 10 \
    --pretrained-bert "bert-base-uncased" \
    --bert-torch-checkpoint "ra4-rsr1_bert-base-uncased_cls.pth.tar" \
    --backend "d3js_graphs" \
    --d3js-host 8000 \
    --docs-limit 500 \
    -o "output" \
    --from-files "<PATH-TO-TEXT-FILE>"
```

Launches server at `http://0.0.0.0:8000/` so you may analyse the results.


<details>
<summary>

## Advanced and Partial Usage
</summary>

### Operations between Graphs

```bash
python3 -m arelight.run.operations --operation SIMILARITY --graph_a emask.json \
  --graph_b jbezos.json --weights y --name elon_SIMILARITY_bezos \
  --description "Similarity between Elon Mask and Jeph Bezos on X/Twitter"
```
![Operations](https://github.com/nicolay-r/ARElight/assets/14871187/90cdbbc8-4a88-4f5f-92a3-355594fa61f0)

### `D3JS`: Launch Graph Builder and DEMO server

Launch Graph Builder for D3JS and (optional) start DEMO server for collections in `output` dir:

```bash
python3 -m arelight.run.infer --backend "d3js_graphs" -o output --d3js-host 8080 
```

### Other languages :ru:

Checkout [wiki-page for greater details](https://github.com/nicolay-r/ARElight/wiki/Low%E2%80%90Resource-Domain-Application)

</details>

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="docs/arekit_logo.png"/></a>
</p>

# ARElight 0.24.0

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.24.0-orange.svg)

### :point_right: [DEMO]() :point_left:

<p align="center">
    <img src="logo.png"/>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities 
in a mass-media texts written in Russian.

### Sentiment Analysis Pipeline

ARElight core is powered by [AREkit](https://github.com/nicolay-r/AREkit) framework,
responsible for raw text sampling.
To annotate objects in text, we use `BERT`-based models trained on
`OntoNotes5` (powered by [DeepPavlov](https://github.com/deeppavlovteam/DeepPavlov))
For relations annotation, we support 
[OpenNRE](https://github.com/thunlp/OpenNRE)
`BERT` models.
The default inference is pretrained BERT with transfer learning based on 
[RuSentRel](https://github.com/nicolay-r/RuSentRel)
and 
[RuAttitudes](https://github.com/nicolay-r/RuAttitudes)
collections, that were sampled and translated into English via 
[arekit-ss](https://github.com/nicolay-r/arekit-ss).


# Installation

```bash
pip install git+https://github.com/nicolay-r/arelight@v0.24.0
```

## Usage

Infer sentiment attitudes from a mass-media document(s).
```bash
python3 -m arelight.run.infer  \
    --sampling-framework "arekit" \
    --ner-model-name "ner_ontonotes_bert_mult" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --terms-per-context 50 \
    --sentence-parser "ru" \
    --text-b-type "nli_m" \
    --tokens-per-context 128 \
    --batch-size 10 \
    --bert-framework "opennre" \
    --pretrained-bert "DeepPavlov/rubert-base-cased" \
    --bert-torch-checkpoint "ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar" \
    --backend "d3js_graphs" \
    --docs-limit 500 \
    -o output/samples \
    --from-files data/texts-inosmi-rus/e0.txt
```

Launches server at `http://0.0.0.0:8000/` so you may analyse the results.

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="docs/arekit_logo.png"/></a>
</p>

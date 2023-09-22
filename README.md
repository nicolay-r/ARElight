# ARElight 0.24.0

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.24.0-orange.svg)

### :point_right: [DEMO](https://github.com/nicolay-r/ARElight/tree/v0.22.0#installation) :point_left:

> **Supported Languages**: Russian

<p align="center">
    <img src="logo.png"/>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities in a mass-media texts written in Russian.

This project is commonly powered by [AREkit](https://github.com/nicolay-r/AREkit) framework.
For Named Entity Recognition in text sentences, 
we adopt [DeepPavlov](https://github.com/deeppavlovteam/DeepPavlov)  (BertOntoNotes model).

# Installation

1. Main library installation
```bash
pip install git+https://github.com/nicolay-r/arelight@v0.24.0
```

2. (Optional) BRAT: [Download](https://github.com/nlplab/brat/releases/tag/v1.3_Crunchy_Frog) 
  and install library, and run standalone server as follows:
```
./install.sh -u
python standalone.py
```

## Usage

Infer sentiment attitudes from a mass-media document(s).
```bash
python3 -m arelight.run.infer --from-files data/texts-inosmi-rus/e1.txt \
    --ner-model-name "ner_ontonotes_bert_mult" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --labels-count 3 \
    --terms-per-context 50 \
    --sentence-parser "ru" \
    --text-b-type "nli_m" \
    --tokens-per-context 128 \
    --bert-framework "opennre" \
    --pretrained-bert "DeepPavlov/rubert-base-cased" \
    --bert-torch-checkpoint "data/ra4-rsr1_DeepPavlov-rubert-base-cased_cls.pth.tar" \
    -o output/brat_inference_output
```
<p align="center">
    <img src="docs/inference-bert-e1.png"/>
</p>

## Reference 

* [Nicolay Rusnachenko: Language Models Application in Sentiment Attitude Extraction Task (2021) [RUS]](https://nicolay-r.github.io/website/data/rusnachenko2021language.pdf)

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="docs/arekit_logo.png"/></a>
</p>

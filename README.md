# ARElight 0.23.0

![](https://img.shields.io/badge/Python-3.6.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.23.0-orange.svg)

### :point_right: [DEMO](#docker-verion-quick) :point_left:

> **Supported Languages**: Russian

<p align="center">
    <img src="logo.png"/>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities in a mass-media texts written in Russian.

This project is commonly powered by [AREkit](https://github.com/nicolay-r/AREkit) framework.
For Named Entity Recognition in text sentences, 
we adopt [DeepPavlov](https://github.com/deeppavlovteam/DeepPavlov)  (BertOntoNotes model).

## Installation

### Docker verion (Quick)

> **Supported Languages**: Russian

> **Other Requirements**: NVidia-docker

* Download [nicolay-r-arelight-0.1.1.tar](https://disk.yandex.ru/d/XXJUXEeaJbqNbA)
* Import container and start Apache hosting: 
```bash
docker import nicolay-r-arelight-0.1.1.tar 
docker run --name arelight -itd --gpus all nicolay-r/arelight:0.1.1
docker attach arelight
service apache2 start
```
* Proceed with BERT demo: http://172.17.0.2/examples/demo/wui_bert.py

> **Supported Languages**: Russian

<p align="center">
    <img src="docs/demo.png"/>
</p>

### Full 
* ARElight:
```bash
# Install the required dependencies
pip install -r dependencies.txt
# Donwload Required Resources
python3.6 download.py
```

* BRAT: [Download](https://github.com/nlplab/brat/releases/tag/v1.3_Crunchy_Frog) 
  and install library, and run standalone server as follows:
```
./install.sh -u
python standalone.py
```

Usage: proceed with the `examples` folder.

## Inference

> **Supported Languages**: Russian

Infer sentiment attitudes from a mass-media document(s).

Using the `BERT` fine-tuned model version:
```bash
python3.6 infer_bert.py --from-files ../data/texts-inosmi-rus/e1.txt \
    --labels-count 3 \
    --terms-per-context 50 \
    --tokens-per-context 128 \
    --text-b-type nli_m \
    --sentence-parser ru \
    -o output/brat_inference_output
```
<p align="center">
    <img src="docs/inference-bert-e1.png"/>
</p>

## Serialization 

> **Supported Languages**: Any

For the `BERT` model:
```bash
python3.6 serialize_bert.py --from-files ../data/texts-inosmi-rus/e1.txt \
    --entities-parser bert-ontonotes \
    --terms-per-context 50 \
    --sentence-parser ru \
    -o output/e1
```

<p align="center">
    <img src="docs/samples-bert.png">
</p>

## Papers

* [Nicolay Rusnachenko: Language Models Application in Sentiment Attitude Extraction Task (2021) [RUS]](https://nicolay-r.github.io/website/data/rusnachenko2021language.pdf)

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="docs/arekit_logo.png"/></a>
</p>

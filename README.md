# ARElight 0.22.0

### :point_right: [DEMO](#docker-verion-quick) :point_left:

This is a DEMO project of sentiment relations annotation, 
commonly powered by [AREkit](https://github.com/nicolay-r/AREkit) framework.

<p align="center">
    <img src="logo.png"/>
</p>


## Dependencies

* arekit == 0.22.0
* gensim == 3.2.0
* deeppavlov == 0.11.0
* rusenttokenize
* brat-v1.3 [[github]](https://github.com/nlplab/brat)
* CUDA

We adopt [DeepPavlov](https://github.com/deepmipt/DeepPavlov) 
for Named Entity Recognition in text sentences (BertOntoNotes model).

# Installation


## Docker verion (Quick)

> NOTE: Requires NVidia-docker

* Download [nicolay-r-arelight-0.1.1.tar](https://drive.google.com/u/0/uc?id=1hd6EOcN5haEJM0dp_sdsNCX36lGG-y8Y&amp;export=download&amp;confirm=t)
* Import container and start Apache hosting: 
```bash
docker import nicolay-r-arelight-0.1.1.tar 
docker run --name arelight -itd --gpus all nicolay-r/arelight:0.1.1
docker attach arelight
service apache2 start
```
* Proceed with BERT demo: http://172.17.0.2/examples/demo/wui_bert.py

![](docs/demo.png)

* PCNN example, finetuned on [RuSentRel](https://github.com/nicolay-r/RuSentRel):
http://172.17.0.2/examples/demo/wui_nn.py

![](docs/demo_pcnn.png)

## Full 
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

# Inference

Infer sentiment attitudes from a mass-media document(s).

Using the `BERT` fine-tuned model version:
```bash
python3.6 infer_texts_bert.py --from-files data/texts-inosmi-rus/e1.txt \
    --labels-count 3 \
    --terms-per-context 50 \
    --tokens-per-context 128 \
    --text-b-type nli_m \
    -o output/brat_inference_output
```
<p align="center">
    <img src="docs/inference-bert-e1.png"/>
</p>

Using the pretrained `PCNN` model (including frames annotation):
```bash
python3.6 infer_texts_nn.py --from-files data/texts-inosmi-rus/e1.txt \
    --model-name pcnn \
    --model-state-dir models/ \
    --terms-per-context 50 \
    --stemmer mystem \
    --entities-parser bert-ontonotes \
    --frames ruattitudes-20 \
    --labels-count 3 \
    --bags-per-minibatch 2 \
    --model-input-type ctx \
    --entity-fmt hidden-simple-eng \
    --emb-filepath data/news_mystem_skipgram_1000_20_2015.bin.gz \
    --synonyms-filepath data/synonyms.txt \
    -o output/brat_inference_output
```

<p align="center">
    <img src="docs/inference-pcnn-e1.png"/>
</p>

# Serialization 

For the `BERT` model:
```bash
python3.6 serialize_texts_bert.py --from-files data/texts-inosmi-rus/e1.txt 
    --entities-parser bert-ontonotes \
    --terms-per-context 50 
```

<p align="center">
    <img src="docs/samples-bert.png">
</p>

For the other neural networks (including embedding and other features):
```bash
python3.6 serialize_texts_nn.py --from-files data/texts-inosmi-rus/e1.txt \
    --entities-parser bert-ontonotes \
    --stemmer mystem \
    --terms-per-context 50 \
    --emb-filepath data/news_mystem_skipgram_1000_20_2015.bin.gz \
    --synonyms-filepath data/synonyms.txt \
    --frames ruattitudes-20 
```

<p align="center">
    <img src="docs/samples-nn.png"/>
</p>

# Other Examples

* Serialize RuSentRel collection for BERT [[code]](examples/serialize_rusentrel_for_bert.py)
* Serialize RuSentRel collection for Neural Networks [[code]](examples/serialize_rusentrel_for_nn.py)
* Finetune BERT on samples [[code]](examples/train_bert.py)
* Finetune Neural Networks on RuSentRel [[code]](examples/train_nn_on_rusentrel.py)

# Papers

* [Nicolay Rusnachenko: Language Models Application in Sentiment Attitude Extraction Task (2021)](https://nicolay-r.github.io/website/data/rusnachenko2021language.pdf)

# Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="docs/arekit_logo.png"/></a>
</p>

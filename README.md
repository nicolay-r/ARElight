# ARElight 0.22.0

This is a DEMO project of sentiment relations annotation, 
commonly powered by [AREkit](https://github.com/nicolay-r/AREkit) framework.

<p align="center">
    <img src="logo.png"/>
</p>


## Dependencies

* arekit == 0.22.0
* gensim == 3.2.0
* deeppavlov == 1.11.0
* rusenttokenize
* brat-v1.3 [[github]](https://github.com/nlplab/brat)

We adopt [DeepPavlov](https://github.com/deepmipt/DeepPavlov) 
for Named Entity Recognition in text sentences (BertOntoNotes model).

# Installation

* Install python related dependencies:
```bash
pip install -r dependencies.txt
```

* [Download](https://github.com/nlplab/brat/releases/tag/v1.3_Crunchy_Frog) 
  and install BRAT library, and run standalone server as follows:
```
./install.sh -u
python standalone.py
```

* Download required resources:
```
python download.py
```

# Inference

<p align="center">
    <img src="docs/inference.png"/>
</p>

> Figure: Named Entities annotation and sentiment attitudes between mentioned named entities.

In order to infer sentiment attitudes, use the `run_test_infer.py` script with the pretrained `PCNN` model:
```bash
python3.6 run_text_infer.py --text "США намерена ввести санкции против Роccии. При этом Москва неоднократно подчеркивала, что ее активность на балтике является ответом именно на действия НАТО и эскалацию враждебного подхода к России вблизи ее восточных границ ..." \
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
    --emb-filepath models/news_mystem_skipgram_1000_20_2015.bin.gz \
    -o data/brat_inference_output.html
```

List of the other input/output files and directories parameters:
```
--synonyms-filepath SYNONYMS_FILEPATH           List of synonyms provided in lines of the source text file.
--vocab-filepath    [VOCAB_FILEPATH]            Custom vocabulary filepath
--emb-npz-filepath  EMBEDDING_MATRIX_FILEPATH   RusVectores embedding filepath
```

List of the other supported parameters is as follows:
```
--model-tag         [MODEL_TAG]                 Optional and additional custom model name suffix. (Default: )
```

# Serialization 

<p align="center">
    <img src="docs/samples.png"/>
</p>

> Figure: The result of samples that might be utilized for ML training in further.

In order to infer sentiment attitudes, use the `run_text_serialize.py` script as follows:
```bash
python3.6 run_text_serialize.py --text "США намерена ввести санкции против Роccии. При этом Москва неоднократно подчеркивала, что ее активность на балтике является ответом именно на действия НАТО и эскалацию враждебного подхода к России вблизи ее восточных границ ..."
    --entities-parser bert-ontonotes \
    --stemmer mystem \
    --terms-per-context 50 \
    --emb-filepath models/news_mystem_skipgram_1000_20_2015.bin.gz \
    --frames ruattitudes-20 
```

List of other supported parameters is as follows:
```
--synonyms-filepath SYNONYMS_FILEPATH           List of synonyms provided in lines of the source text file.
```

# Training other models

[Please proceed with the following Readme/Tutorial](README_train_custom_model.md)

# Powered by

AREkit [[github]](https://github.com/nicolay-r/AREkit)
<p align="left">
    <img src="docs/arekit_logo.png"/>
</p>

DeepPavlov [[github]](https://github.com/deepmipt/DeepPavlov)
<p align="left">
    <img src="docs/deeppavlov_logo.png"/>
</p>

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

# Inference

<p align="center">
    <img src="docs/inference.png"/>
</p>

> Figure: Named Entities annotation and sentiment attitudes between mentioned named entities.

In order to infer sentiment attitudes, use the `run_test_infer.py` script with the pretrained `PCNN` model:
```bash
python3.6 run_text_infer.py --text "США намерена ввести санкции против Роccии. При этом Москва неоднократно подчеркивала, что ее активность на балтике является ответом именно на действия НАТО и эскалацию враждебного подхода к России вблизи ее восточных границ ..." \
    --model-name pcnn \
    --model-state-dir models/ 
    -o data/brat_inference_output.html
```

List of the other input/output files and directories parameters:
```
--emb-filepath      EMBEDDING_FILEPATH          RusVectores embedding filepath
--synonyms-filepath SYNONYMS_FILEPATH           List of synonyms provided in lines of the source text file.
--vocab-filepath    [VOCAB_FILEPATH]            Custom vocabulary filepath
--emb-npz-filepath  EMBEDDING_MATRIX_FILEPATH   RusVectores embedding filepath
--frames            [FRAMES]                    Collection for frames annotation in text (Default: ruattitudes-20)
```

List of the supported parameters is as follows:
```
--model-name {cnn,att-cnn,att-ef-cnn,att-se-cnn,att-se-pcnn,
              att-se-bilstm,att-sef-cnn,att-sef-pcnn,att-sef-bilstm,
              att-ef-pcnn,att-ef-bilstm,att-pcnn,att-frames-cnn,
              att-frames-pcnn,self-att-bilstm,bilstm,ian,ian-ends,ian-ef,
              ian-se,ian-sef,pcnn,rnn,rcnn,rcnn-att-p-zhou,rcnn-att-z-yang,
              att-frames-bilstm,att-bilstm-z-yang,att-bilstm} 
              Name of a model to be utilized in experiment

--labels-count      LABELS_COUNT                Labels count in an output classifier
--bags-per-minibatch [BAGS_PER_MINIBATCH]       Bags per minibatch count (Default: 2)
--model-tag         [MODEL_TAG]                 Optional and additional custom model name suffix. (Default: )
--model-input-type  [{ctx,mi-mp,mi-self-att}]   Input format type (Default: ModelInputType.SingleInstance)
--entity-fmt        {rus-cased-fmt,rus-simple,simple-uppercase,simple,sharp-simple}
                    Entity formatter type
--entities-parser   {no,bert-ontonotes}         Adopt entities parser in text processing (default: bert-ontonotes)
--stemmer           [{mystem}]                  Stemmer (Default: mystem)
--terms-per-context [TERMS_PER_CONTEXT]
                    The max possible length of an input context in terms
                    (Default: 50) NOTE: Use greater or equal value for
                    this parameter during experimentprocess; otherwise you
                    may encounter with exception during sample creation
                    process!
```

# Serialization 

<p align="center">
    <img src="docs/samples.png"/>
</p>

> Figure: The result of samples that might be utilized for ML training in further.

In order to infer sentiment attitudes, use the `run_text_serialize.py` script as follows:
```bash
python3.6 run_text_serialize.py --text "США намерена ввести санкции против Роccии. При этом Москва неоднократно подчеркивала, что ее активность на балтике является ответом именно на действия НАТО и эскалацию враждебного подхода к России вблизи ее восточных границ ..."
```

List of the supported parameters is as follows:
```
--entities-parser   {no,bert-ontonotes}         Adopt entities parser in text processing (default: bert-ontonotes)
--emb-filepath      EMBEDDING_FILEPATH          RusVectores embedding filepath
--terms-per-context [TERMS_PER_CONTEXT]         The max possible length of an input context in terms
                                                (Default: 50) NOTE: Use greater or equal value for
                                                this parameter during experimentprocess; otherwise you
                                                may encounter with exception during sample creation
                                                process!
--entity-fmt        {rus-cased-fmt,rus-simple,simple-uppercase,simple,sharp-simple}
                                                Entity formatter type
--stemmer           [{mystem}]                  Stemmer (Default: mystem)
--synonyms-filepath SYNONYMS_FILEPATH           List of synonyms provided in lines of the source text file.
--frames            [FRAMES]                    Collection for frames annotation in text (Default: ruattitudes-20)
```

# Powered by

AREkit [[github]](https://github.com/nicolay-r/AREkit)
<p align="left">
    <img src="docs/arekit_logo.png"/>
</p>

DeepPavlov [[github]](https://github.com/deepmipt/DeepPavlov)
<p align="left">
    <img src="docs/deeppavlov_logo.png"/>
</p>

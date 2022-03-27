# Custom Model Training Tutorial

1. Prepare data
2. Train model

# 1. Large Data Serialization 
> **NOTE:** Provided for [RuSentRel](https://github.com/nicolay-r/RuSentRel) collection

```bash
python3.6 run_rusentrel_serialize.py
```

List of the parameters is as follows:
```
--emb-filepath      EMBEDDING_FILEPATH RusVectores embedding filepath
--terms-per-context [TERMS_PER_CONTEXT]
                    The max possible length of an input context in terms
                    (Default: 50) NOTE: Use greater or equal value for
                    this parameter during experimentprocess; otherwise you
                    may encounter with exception during sample creation
                    process!
--entity-fmt        {rus-cased-fmt,rus-simple,simple-uppercase,simple,sharp-simple}
                    Entity formatter type
--stemmer           [{mystem}]  Stemmer (Default: mystem)
--labels-count      LABELS_COUNT Labels count in an output classifier
--balance-samples   BALANCE_SAMPLES
                    Use balancing for Train type during sample
                    serialization process"
--dist-between-att-ends [DIST_BETWEEN_ENDS]
                    Distance in terms between attitude participants in
                    terms.(Default: None)
```

# 2. Large Data Training
> **Limitation #1:** Provided for [RuSentRel](https://github.com/nicolay-r/RuSentRel) 
> collection (in terms of paths only!)

> **Limitation #2:** Support Tensorflow-based neural networks.

```bash
python3.6 run_rusentrel_train.py
```

List of the parameters related to models:
```bash
--labels-count      LABELS_COUNT            Labels count in an output classifier
--bags-per-minibatch [BAGS_PER_MINIBATCH]   Bags per minibatch count (Default: 2)
--model-name        {cnn,att-cnn,att-ef-cnn,att-se-cnn,att-se-pcnn,att-se-bilstm,att-sef-cnn,att-sef-pcnn,att-sef-bilstm,att-ef-pcnn,att-ef-bilstm,att-pcnn,att-frames-cnn,att-frames-pcnn,self-att-bilstm,bilstm,ian,ian-ends,ian-ef,ian-se,ian-sef,pcnn,rnn,rcnn,rcnn-att-p-zhou,rcnn-att-z-yang,att-frames-bilstm,att-bilstm-z-yang,att-bilstm}
                    Name of a model to be utilized in experiment
--model-input-type [{ctx,mi-mp,mi-self-att}] Input format type (Default: ModelInputType.SingleInstance)
--dropout-keep-prob [DROPOUT_KEEP_PROB]     Dropout keep prob (Default: 0.5)
--learning-rate     [LEARNING_RATE]         Learning Rate (Default: 0.1) 
--epochs            [EPOCHS]                Epochs count (Default: 150)
--model-state-dir   [MODEL_LOAD_DIR]        Use pretrained state as initial
```

List of parameters related to input post-processing:
```
--terms-per-context [TERMS_PER_CONTEXT]
                    The max possible length of an input context in terms
                    (Default: 50) NOTE: Use greater or equal value for
                    this parameter during experimentprocess; otherwise you
                    may encounter with exception during sample creation
                    process!
--dist-between-att-ends [DIST_BETWEEN_ENDS]
                    Distance in terms between attitude participants in
                    terms.(Default: None)
```

List of the parameters related to embeddings:
```
--vocab-filepath    [VOCAB_FILEPATH]            Custom vocabulary filepath
--emb-npz-filepath  EMBEDDING_MATRIX_FILEPATH   RusVectores embedding filepath
```

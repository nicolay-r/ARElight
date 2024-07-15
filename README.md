# ARElight 0.25.0

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.25.1-orange.svg)
[![](https://img.shields.io/badge/demo-0.24.0-purple.svg)](https://guardeec.github.io/arelight_demo/template.html)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ARElight/blob/v0.24.0/ARElight.ipynb)

### :point_right: [DEMO](https://guardeec.github.io/arelight_demo/template.html) :point_left:

<p align="center">
    <img src="logo.png"/>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities 
in texts. 
This repository is a part of the **ECIR-2024** demo paper: 
[ARElight: Context Sampling of Large Texts for Deep Learning Relation Extraction](https://link.springer.com/chapter/10.1007/978-3-031-56069-9_23).


# Installation

```bash
pip install git+https://github.com/nicolay-r/arelight@v0.24.1
```

## Usage: Inference
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nicolay-r/ARElight/blob/v0.24.0/ARElight.ipynb)

Infer sentiment attitudes from text file **in English**:
```bash
python3 -m arelight.run.infer  \
    --sampling-framework "arekit" \
    --ner-framework "deeppavlov" \
    --ner-model-name "ner_ontonotes_bert" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --terms-per-context 50 \
    --sentence-parser "nltk:english" \
    --tokens-per-context 128 \
    --bert-framework "opennre" \
    --batch-size 10 \
    --pretrained-bert "bert-base-cased" \
    --bert-torch-checkpoint "ra4-rsr1_bert-base-cased_cls.pth.tar" \
    --backend "d3js_graphs" \
    --docs-limit 500 \
    -o "output" \
    --from-files "<PATH-TO-TEXT-FILE>"
```

> **NOTE:** [Applying ARElight for **non-english texts**](https://github.com/nicolay-r/ARElight/wiki/Language-Specific-Application)
>

<details>
<summary>

### Parameters
</summary>

The complete documentation is avalable via `-h` flag:
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
* `bert-framework` -- samples classification framework; we support [[OpenNRE]](https://github.com/thunlp/OpenNRE).
    * `text-b-type` -- (optional) `NLI` or None [[supported]](https://github.com/nicolay-r/ARElight/blob/a17088a98729e3092de1666bef9ba8327ef30b80/arelight/samplers/bert.py#L14).
    * `pretrained-bert` -- pretrained state name.
    * `batch-size` -- amount of samples per single inference iteration.
    * `tokens-per-context` -- size of input.
    * `bert-torch-checkpoint` -- fine-tuned state.
    * `device-type` -- `cpu` or `gpu`.
    * `labels-fmt` -- list of the mappings from `label` to integer value; is a `p:1,n:2,u:0` by default, where:
        * `p` -- positive label, which is mapped to `1`.
        * `n` -- negative label, which is mapped to `2`.
        * `u` -- undefined label (optional), which is mapped to `0`.
* `backend` -- type of the backend (`d3js_graphs` by default).
    * `host` -- port on which we expect to launch localhost server.
    * `label-names` -- default mapping is `p:pos,n:neg,u:neu`.
* `-o` -- output folder for result collections and demo.

Framework parameters mentioned above as well as their related setups might be ommited.

</details>

To Launch Graph Builder for D3JS and (optional) start DEMO server for collections in `output` dir:

```bash
cd output && python -m http.server 8000
```
 
## Usage: Graph Operations

For graph analysis you can perform several graph operations by this script:

1. Arguments mode:

```bash
python3 -m arelight.run.operations \
	--operation "<OPERATION-NAME>" \
	--graph_a_file output/force/boris.json \
  	--graph_b_file output/force/rishi.json \
  	--weights y \
  	-o output \
  	--description "[OPERATION] between Boris Johnson and Rishi Sunak on X/Twitter"
```

2. Interactive mode:

```bash
python3 -m arelight.run.operations
```

`arelight.run.operations` allows you to operate ARElight's outputs using graphs: you can merge graphs, find their similarities or differences.


<details>
<summary>

### Parameters

</summary>

* `--graph_a_file` and `--graph_b_file` are used to specify the paths to the `.json` files for graphs A and B, which are used in the operations.
  These files should be located in the `<your_output/force>` folder.
* `--name` -- name of the new graph.
* `--description` -- description of the new graph.
* `--host` -- determines the server port to host after the calculations.
* `-o` -- option allows you to specify the path to the folder where you want to store the output.
  You can either create a new output folder or use an existing one that has been created by ARElight.

</details>

<details>
<summary>

### Parameter `operation`
</summary>

#### Preparation

Consider that you used ARElight script for X/Twitter 
to [infer relations](#usage-inference) from
messages of UK politicians `Boris Johnson` and `Rishi Sunak`:

```bash
python3 -m arelight.run.infer ...other arguments... \
	-o output --collection-name "boris" --from-files "twitter_boris.txt"
	
python3 -m arelight.run.infer  ...other arguments... \
	-o output --collection-name "rishi" --from-files "twitter_rishi.txt"
```
According to the [results section](#layout-of-the-files-in-output), you will have `output` directory with 2 files `force` layout graphs:
```lua
output/
├── force/
    ├──  rishi.json
    └──  boris.json
```

#### List of Operations

You can do the following operations to combine several outputs, ot better understand similarities, and differences between them:

**UNION** $(G_1 \cup G_2)$ - combine multiple graphs together.
* The result graph contains all the vertices and edges that are in $G_1$ and $G_2$. 
The edge weight is given by $W_e = W_{e1} + W_{e2}$, and the vertex weight is its weighted degree centrality: $W_v = \sum_{e \in E_v} W_e(e)$.
  ```bash
  python3 -m arelight.run.operations --operation UNION \
      --graph_a_file output/force/boris.json \
      --graph_b_file output/force/rishi.json \
      --weights y -o output --name boris_UNION_rishi \
      --description "UNION of Boris Johnson and Rishi Sunak Twits"
  ```
  ![union](https://github.com/nicolay-r/ARElight/assets/14871187/eaac6758-69f7-4cc1-a631-7ce132757b29)

**INTERSECTION** $(G_1 \cap G_2)$ - what is similar between 2 graphs?
* The result graph contains only the vertices and edges common to $G_1$ and $G_2$. 
The edge weight is given by $W_e = \min(W_{e1},W_{e2})$, and the vertex weight is its weighted degree centrality: $W_v = \sum_{e \in E_v} W_e(e)$.
  ```bash
  python3 -m arelight.run.operations --operation INTERSECTION \
      --graph_a_file output/force/boris.json \
      --graph_b_file output/force/rishi.json \
      --weights y -o output --name boris_INTERSECTION_rishi \
      --description "INTERSECTION between Twits of Boris Johnson and Rishi Sunak"
  ```
  ![intersection](https://github.com/nicolay-r/ARElight/assets/14871187/286bd1ce-dbb0-4370-bfbe-245330ae6204)


**DIFFERENCE** $(G_1 - G_2)$ - what is unique in one graph, that another graph doesn't have? 

* **NOTE:** this operation is not commutative $(G_1 - G_2) ≠ G_2 - G_1)$)_
* The results graph contains all the vertices from $G_1$ but only includes edges from $E_1$ that either don't appear in $E_2$ or have larger weights in $G_1$ compared to $G_2$. 
The edge weight is given by $W_e = W_{e1} - W_{e2}$ if $e \in E_1$, $e \in E_1 \cap E_2$ and $W_{e1}(e) > W_{e2}(e)$.
  ```bash
  python3 -m arelight.run.operations --operation DIFFERENCE \
      --graph_a_file output/force/boris.json \
      --graph_b_file output/force/rishi.json \
      --weights y -o output --name boris_DIFFERENCE_rishi \
      --description "Difference between Twits of Boris Johnson and Rishi Sunak"
  ```
  ![difference](https://github.com/nicolay-r/ARElight/assets/14871187/8b036ce6-6607-4588-b0cf-4704647f55ff)

</details>

<details>
<summary>

### Parameter `weights`
</summary>

You have the option to specify whether to include edge weights in calculations or not. 
These weights represent the frequencies of discovered edges, indicating how often a relation between two instances was found in the text analyzed by ARElight.
* `--weights`
  * `y`: the result will be based on the union, intersection, or difference of these frequencies.
  * `n`: all weights of input graphs will be set to 1. In this case, the result will reflect the union, intersection, or difference of the graph topologies, regardless of the frequencies. This can be useful when the existence of relations is more important to you, and the number of times they appear in the text is not a significant factor.
  > Note that using or not using the `weights` option may yield different topologies:
  > 
  ![weights](https://github.com/nicolay-r/ARElight/assets/14871187/43ad2054-d413-47ee-ac8b-d06af6921214)

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
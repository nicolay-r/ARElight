# ARElight 0.24.0

![](https://img.shields.io/badge/Python-3.9-brightgreen.svg)
![](https://img.shields.io/badge/AREkit-0.24.0-orange.svg)
[![](https://img.shields.io/badge/demo-0.24.0-purple.svg)](https://guardeec.github.io/arelight_demo/template.html)

### :point_right: [DEMO](https://guardeec.github.io/arelight_demo/template.html) :point_left:

<p align="center">
    <img src="logo.png"/>
</p>

ARElight is an application for a granular view onto sentiments between mentioned named entities 
in texts.


# Installation

```bash
pip install git+https://github.com/nicolay-r/arelight@v0.24.0
```

## Usage: Inference

Infer sentiment attitudes from text file **in English**:
```bash
python3 -m arelight.run.infer  \
    --sampling-framework "arekit" \
    --ner-model-name "ner_ontonotes_bert" \
    --ner-types "ORG|PERSON|LOC|GPE" \
    --terms-per-context 50 \
    --sentence-parser "nltk_en" \
    --tokens-per-context 128 \
    --bert-framework "opennre" \
    --batch-size 10 \
    --pretrained-bert "bert-base-cased" \
    --bert-torch-checkpoint "ra4-rsr1_bert-base-cased_cls.pth.tar" \
    --backend "d3js_graphs" \
    --d3js-host 8000 \
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
    * `text` -- textual content of the single document. 
    * `from-files` -- list of filepaths to the related documents.
      * for the `.csv` files we consider that each line of the particular `column` as a separated document.
        * `csv-sep` -- separator between columns.
        * `csv-column` -- name of the column in CSV file.
    * `collection-name` -- name of the result files based on sampled documents.
    * `terms-per-context` -- total amount of words for a single sample.
    * `sentence-parser` -- parser utilized for document split into sentences; list of the [[supported parsers]](https://github.com/nicolay-r/ARElight/blob/a17088a98729e3092de1666bef9ba8327ef30b80/arelight/run/utils.py#L15).
    * `synonyms-filepath` -- text file with listed synonymous entries, grouped by lines. [[example]](https://github.com/nicolay-r/RuSentRel/blob/master/synonyms.txt).
    * `stemmer` -- for words lemmatization (optional); we support [[PyMystem]](https://pypi.org/project/pymystem3/).
    * `ner-framework` -- type of the Named Entity Recognition framework; we support [[DeepPavlov]](https://github.com/deeppavlov/DeepPavlov).
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
* `backend` -- type of the backend (`d3js_graphs` by default).
    * `d3js-host` -- port on which we expect to launch localhost server.
* `-o` -- output folder for result collections and demo.

Framework parameters mentioned above as well as their related setups might be ommited.
For example, to Launch Graph Builder for D3JS and (optional) start DEMO server for collections in `output` dir:

```bash
python3 -m arelight.run.infer --backend "d3js_graphs" -o output --d3js-host 8080 
```

</details>
 
Launches server at `http://0.0.0.0:8000/` so you may analyse the results. 

[![](https://img.shields.io/badge/demo-0.24.0-purple.svg)](https://guardeec.github.io/arelight_demo/template.html)

![image](https://github.com/nicolay-r/ARElight/assets/14871187/341f3b51-d639-46b6-83fe-99b542b1751b)


## Usage: Graph Operations

For graph analysis you can perform several graph operations by this script:

With arguments mode:

```bash
python3 -m arelight.run.operations \
	--operation SIMILARITY \
	--graph_a_file output/force/boris.json \
  	--graph_b_file output/force/rishi.json \
  	--weights y \
  	-o output \
  	--description "Similarity between Boris Johnson and Rishi Sunak on X/Twitter"
```

Interactive mode (do not pass any arguments):

```bash
python3 -m arelight.run.operations
```

`arelight.run.operations` allows you to operate ARElight's outputs using graphs: you can merge graphs, find their similarities or differences.

<details>
<summary>

### Parameter `operation`
</summary>

You can do the following operations to combine several outputs, ot better understand similarities, and differences between them:

**UNION** $(G_1 \cup G_2)$ - combine multiple graphs together.

Here, $G$ contains all the vertices and edges that are in $G_1$ and $G_2$. The edge weight is given by $W_e = W_{e1} + W_{e2}$, and the vertex weight is its weighted degree centrality: $W_v = \sum_{e \in E_v} W_e(e)$.

Helps to unite several graphs, e.g. imagine that you used ARElight script for Twits messages of UK politicians Boris Johnson and Rishi Sunak:

```bash
python3 -m arelight.run.infer 
	...other arguments... 
	-o output \ 
	--name boris \
	--from-files "twitter_boris.txt"
	
python3 -m arelight.run.infer \
	...other arguments... 
	-o output \
	--name rishi \
	--from-files "twitter_rishi.txt"
```
, so now you have folder `output` with 2 files: (1)`output/radial/rishi.json`, (2)`output/radial/boris.json`.

You can run operation UNION to create a single graph that describe Twits of of them both:

```bash
python3 -m arelight.run.operations --operation UNION \
	--graph_a_file output/force/boris.json \
  	--graph_b_file output/force/rishi.json \
  	--weights y \
  	-o output \
  	--name boris_AND_rishi \
  	--description "Twits of Boris Johnson and Rishi Sunak"
```
![operations](https://drive.google.com/uc?export=view&id=16PoDg_4AM9Z1l2ZGbL15lknG6wg39EMA)

**INTERSECTION** $(G_1 \cap G_2)$ - what is similar between 2 graphs?

In this operation, $G$ contains only the vertices and edges common to $G_1$ and $G_2$. The edge weight is given by $W_e = \min(W_{e1},W_{e2})$, and the vertex weight is its weighted degree centrality: $W_v = \sum_{e \in E_v} W_e(e)$.

Helps to extract what is similar, e.g. you have the same folder `output` with 2 files: (1)`output/radial/rishi.json`, (2)`output/radial/boris.json`. You can run operation INTERSECTION to create graph that describe what is similar between Twits of Rishi Sunak and Boris Johnson:

```bash
python3 -m arelight.run.operations --operation INTERSECTION \
	--graph_a_file output/force/boris.json \
  	--graph_b_file output/force/rishi.json \
  	--weights y \
  	-o output \
  	--name boris_SIMILARITY_rishi \
  	--description "Similarity between Twits of Boris Johnson and Rishi Sunak"
```

![operations](https://drive.google.com/uc?export=view&id=17emwHJ-7Tb_ISnTkDwWCd5pxmu8IjqUg)

**DIFFERENCE** $(G_1 - G_2)$ - what is unique in one graph, that another graph doesn't have? 

_(note: this operation is not commutative $(G_1 - G_2) ≠ G_2 - G_1)$)_

$G$ contains all the vertices from $G_1$ but only includes edges from $E_1$ that either don't appear in $E_2$ or have larger weights in $G_1$ compared to $G_2$. The edge weight is given by $W_e = W_{e1} - W_{e2}$ if $e \in E_1$, $e \in E_1 \cap E_2$ and $W_{e1}(e) > W_{e2}(e)$.

Helps to extract what is unique, e.g.: you have the same folder `output` with 2 files: (1)`output/radial/rishi.json`, (2)`output/radial/boris.json`. You can run operation DIFFERENCE to create graph that describe what is unique in Twits of Boris Johnson in comparison to Rishi Sunak:

```bash
python3 -m arelight.run.operations --operation DIFFERENCE \
	--graph_a_file output/force/boris.json \
  	--graph_b_file output/force/rishi.json \
  	--weights y \
  	-o output \
  	--name boris_DIFFERENCE_rishi \
  	--description "Difference between Twits of Boris Johnson and Rishi Sunak"
  	--vis y
```
![operations](https://drive.google.com/uc?export=view&id=109PmaZeWYtEUTPEX-DXvIYcav5r39rrn)
</details>

<details>
<summary>

### Parameter `weights`
</summary>

You have the option to specify whether to include edge weights in calculations or not. These weights represent the frequencies of discovered edges, indicating how often a relation between two instances was found in the text analyzed by ARElight.

When you set the flag to ```--weights y```, the result will be based on the union, intersection, or difference of these frequencies.

When you set the flag to ```--weights n```, all weights of input graphs will be set to 1. In this case, the result will reflect the union, intersection, or difference of the graph topologies, regardless of the frequencies. This can be useful when the existence of relations is more important to you, and the number of times they appear in the text is not a significant factor.

Note that using or not using the ```--weights``` option may yield different topologies:
![operations](https://drive.google.com/uc?export=view&id=1xPlV8LwY28l00ZVoS3URHc1MSONQDhuf)

</details>

<details>
<summary>

### Parameters (Others)

</summary>


* `--graph_a_file` and `--graph_b_file` are used to specify the paths to the `.json` files for graphs A and B, which are used in the operations. These files should be located in the `<your_output/radial>` folder.
* `-o` option allows you to specify the path to the folder where you want to store the output. You can either create a new output folder or use an existing one that has been created by ARElight.
* `--name` and `--description` options, you can provide a name for the resulting `.json` file and a description for it.
* `--host` -- determines whether to run the visualization server after the calculations. You can choose ```y``` for yes or ```n``` for no.

</details>



## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>

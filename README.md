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

## Usage

Infer sentiment attitudes from text file:
```bash
python3 -m arelight.run.infer  \
    --sampling-framework "arekit" \
    --ner-model-name "ner_ontonotes_bert_mult" \
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

> **NOTE:** [Details for ARElight launching for **non-english texts**](https://github.com/nicolay-r/ARElight/wiki/Language-Specific-Application))
 
Launches server at `http://0.0.0.0:8000/` so you may analyse the results. 

[![](https://img.shields.io/badge/demo-0.24.0-purple.svg)](https://guardeec.github.io/arelight_demo/template.html)

![image](https://github.com/nicolay-r/ARElight/assets/14871187/341f3b51-d639-46b6-83fe-99b542b1751b)

<details>
<summary>

## Advanced and Partial Usage
</summary>

### Operations between Graphs

For graph analysis you can perform several graph operations by this script:

```bash
python3 -m arelight.run.operations --operation SIMILARITY --graph_a emask.json \
  --graph_b jbezos.json --weights y --name elon_SIMILARITY_bezos \
  --description "Similarity between Elon Mask and Jeph Bezos on X/Twitter"
```

**Union** $(G_1 \cup G_2)$.
Helps to unite several graphs, e.g. graphs of multiple social network users.
Here, $G$ contains all the vertices and edges that are in $G_1$ and $G_2$. The edge weight is given by $W_e = W_{e1} + W_{e2}$, and the vertex weight is its weighted degree centrality: $W_v = \sum_{e \in E_v} W_e(e)$.

**Intersection** $(G_1 \cap G_2)$.
Helps to extract what is similar, e.g. what is similar between two social network users.
In this operation, $G$ contains only the vertices and edges common to $G_1$ and $G_2$. The edge weight is given by $W_e = \min(W_{e1},W_{e2})$, and the vertex weight is its weighted degree centrality: $W_v = \sum_{e \in E_v} W_e(e)$.

**Difference** $(G_1 - G_2)$.
Helps to extract what is unique, e.g. what is unique in graph of user A in comparison to user B (note: this operation is not commutative).
$G$ contains all the vertices from $G_1$ but only includes edges from $E_1$ that either don't appear in $E_2$ or have larger weights in $G_1$ compared to $G_2$. The edge weight is given by $W_e = W_{e1} - W_{e2}$ if $e \in E_1$, $e \in E_1 \cap E_2$ and $W_{e1}(e) > W_{e2}(e)$.

![operations](https://github.com/nicolay-r/ARElight/assets/14871187/c0e6e8c9-a037-49b0-9404-86edbebf2a23)

### `D3JS`: Launch Graph Builder and DEMO server

Launch Graph Builder for D3JS and (optional) start DEMO server for collections in `output` dir:

```bash
python3 -m arelight.run.infer --backend "d3js_graphs" -o output --d3js-host 8080 
```

</details>

## Powered by

* AREkit [[github]](https://github.com/nicolay-r/AREkit)

<p float="left">
<a href="https://github.com/nicolay-r/AREkit"><img src="https://github.com/nicolay-r/ARElight/assets/14871187/01232f7a-970f-416c-b7a4-1cda48506afe"/></a>
</p>

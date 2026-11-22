#!/bin/bash
export PYTHONPATH=..
python3 -m arelight.run.infer \
	--sampling-framework "arekit" \
	--translate-provider "gt_402.py" \
	--translate-text "auto:ru" \
	--ner-provider "dp_130.py" \
	--ner-model-name "ner_ontonotes_bert_mult" \
	--ner-types "ORG|PERSON|LOC|GPE" \
	--terms-per-context 50 \
	--docs-limit 10 \
	--sentence-parser "nltk:english" \
	-o "output" \
	--from-files $1 \
	--log-file "arelight.log.txt" \
    --backend "d3js_graphs" \
    --inference-writer "sqlite3" \
    --inference-filename "replicate_104.py" \
    --inference-model-name "meta/meta-llama-3-70b-instruct"

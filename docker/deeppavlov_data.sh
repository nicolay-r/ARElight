#!/bin/bash

# This is required since DeepPavlov downloader might not work.

# Apache server demo dir.
demo_dir="/var/www/demo/"
dp_dir=$demo_dir.deeppavlov

# Assigning remote URLs.
ner_model_url="http://files.deeppavlov.ai/deeppavlov_data/ner_ontonotes_bert_mult_v1.tar.gz"
bert_model_url="http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip"

# Donwload models.
wget $ner_model_url
wget $bert_model_url

# Unpacking models.
tar -xvf ner_ontonotes_bert_mult_v1.tar.gz
unzip multi_cased_L-12_H-768_A-12.zip
rm ner_ontonotes_bert_mult_v1.tar.gz
rm multi_cased_L-12_H-768_A-12.zip

# Prepare target folders.
mkdir -p "$dp_dir/models/"
mkdir -p "$dp_dir/downloads/bert_models/"

# Moving models.
mv "ner_ontonotes_bert_mult" "$dp_dir/models/"
mv "multi_cased_L-12_H-768_A-12" "$dp_dir/downloads/bert_models/"

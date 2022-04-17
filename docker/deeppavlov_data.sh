# This is required since DeepPavlov downloader might not work.

# Apache server demo dir.
demo_dir="/var/www/demo/"

# Assigning remote URLs.
ner_model_url="http://files.deeppavlov.ai/deeppavlov_data/ner_ontonotes_bert_mult_v1.tar.gz"
bert_model_url="http://files.deeppavlov.ai/deeppavlov_data/bert/multi_cased_L-12_H-768_A-12.zip"

# Donwload models.
wget $(ner_model_url)
wget $(bert_model_url)

# Unpacking models.
tar -xvf ner_ontonotes_bert_mult_v1.tar.gz
unzip multi_cased_L-12_H-768_A-12.zip
rm ner_ontonotes_bert_mult_v1
rm multi_cased_L-12_H-768_A-12

# Prepare target folders.
mkdir -p "$(demo_dir).deeppavlov/models/"
mkdir -p "$(demo_dir).deeppavlov/downloads/bert_models/"

# Moving models.
mv "ner_ontonotes_bert_mult_v1" "$(demo_dir).deeppavlov/models/"
mv "multi_cased_L-12_H-768_A-12" "$(demo_dir).deeppavlov/downloads/bert_models/"

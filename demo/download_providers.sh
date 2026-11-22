#!/bin/sh
# Download the NER, translation, and LLM providers used by the inference example.

set -eu

cd "$(dirname "$0")"

curl -fsSL -o dp_130.py \
    https://raw.githubusercontent.com/nicolay-r/nlp-thirdgate/master/ner/dp_130.py
curl -fsSL -o replicate_104.py \
    https://raw.githubusercontent.com/nicolay-r/nlp-thirdgate/master/llm/replicate_104.py
curl -fsSL -o googletrans_402.py \
    https://raw.githubusercontent.com/nicolay-r/nlp-thirdgate/master/text-translation/googletrans_402.py

printf '%s\n' "$(pwd)/dp_130.py" "$(pwd)/replicate_104.py" "$(pwd)/googletrans_402.py"

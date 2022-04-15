#!/bin/bash

script_dir=$(dirname $0)
mkdir --parents $script_dir/arelight
rsync -r $script_dir/../setup.py $script_dir/arelight
rsync -r $script_dir/../dependencies.txt $script_dir/arelight/
rsync -r $script_dir/../download.py $script_dir/arelight
rsync -r $script_dir/../arelight/ $script_dir/arelight/arelight/
rsync -r $script_dir/../examples/ $script_dir/arelight/examples/
rsync -r $script_dir/../examples/demo/ $script_dir/demo/

# Download brat
curl https://codeload.github.com/nlplab/brat/zip/refs/heads/v1.3p1 --output brat.zip
unzip brat.zip && mv brat-1.3p1 brat
rm brat.zip

docker build -t nicolay-r/arelight $script_dir
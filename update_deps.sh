#!/bin/bash
pip3 uninstall arekit
pip3 install git+https://github.com/nicolay-r/AREkit@0.25.2-rc --no-deps
pip3 uninstall bulk_ner
pip3 install git+https://github.com/nicolay-r/bulk-ner@main
pip3 uninstall bulk_translate
pip3 install git+https://github.com/nicolay-r/bulk-translate@master
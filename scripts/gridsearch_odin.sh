#!/usr/bin/bash

python3 grid_search.py --model DenseNet_augm --supervisor odin --subset 1000
python3 grid_search.py --model VGG_augm --supervisor odin --subset 1000
python3 grid_search.py --model WRN28_augm --supervisor odin --subset 1000
python3 grid_search.py --model WRN40_augm --supervisor odin --subset 1000


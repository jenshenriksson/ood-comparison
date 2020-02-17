#!/usr/bin/bash

python3 grid_search_openmax.py --model DenseNet_augm --supervisor openmax --subset 1000
python3 grid_search_openmax.py --model VGG_augm --supervisor openmax --subset 1000
python3 grid_search_openmax.py --model WRN28_augm --supervisor openmax --subset 1000
python3 grid_search_openmax.py --model WRN40_augm --supervisor openmax --subset 1000


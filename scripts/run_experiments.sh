#!/usr/bin/bash

python3 run_saved_epochs.py --model DenseNet_augm --supervisor $1 --outliers $2 --subset $3
python3 run_saved_epochs.py --model VGG_augm --supervisor $1 --outliers $2 --subset $3
python3 run_saved_epochs.py --model WRN28_augm --supervisor $1 --outliers $2 --subset $3
python3 run_saved_epochs.py --model WRN40_augm --supervisor $1 --outliers $2 --subset $3


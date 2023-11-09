#!/bin/bash

PYTHON_SCRIPT="src/utils/make_dataset.py"

if [ ! -d "mixtures_data/train_clean" ]; then
    python $PYTHON_SCRIPT --part train-clean-100 --path_mixtures mixtures_data/train_clean --num_speakers 240 --nfiles 13000
else
    echo "Directory mixtures_data/train_clean already exists. Skipping..."
fi

if [ ! -d "mixtures_data/test_clean" ]; then
    python $PYTHON_SCRIPT --part test-clean --path_mixtures mixtures_data/test_clean --test True --nfiles 1000
else
    echo "Directory mixtures_data/test_clean already exists. Skipping..."
fi

if [ ! -d "mixtures_data/dev_clean" ]; then
    python $PYTHON_SCRIPT --part dev-clean --path_mixtures mixtures_data/dev_clean --nfiles 100
else
    echo "Directory mixtures_data/dev_clean already exists. Skipping..."
fi

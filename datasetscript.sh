#!/bin/bash

PYTHON_SCRIPT="src/utils/make_dataset.py"

if [ ! -d "mixtures_data/train_clean" ]; then
    python $PYTHON_SCRIPT --part train-clean-100 --path_mixtures mixtures_data/train_clean --num_speakers 96 --nfiles 7000
else
    echo "Directory mixtures_data/train_clean already exists. Skipping..."
fi

if [ ! -d "mixtures_data/test_clean" ]; then
    python $PYTHON_SCRIPT --part test-clean --path_mixtures mixtures_data/test_clean --test True
else
    echo "Directory mixtures_data/test_clean already exists. Skipping..."
fi

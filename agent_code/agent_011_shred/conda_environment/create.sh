#!/usr/bin/env bash

# conda create -n bomberman_rl python=3.6
conda update -n base -c defaults conda
conda env create -f environment.yml
conda info --envs
# conda list

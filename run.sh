#!/bin/bash

cd src
python3 prepare_data.py sample
python3 train.py sample
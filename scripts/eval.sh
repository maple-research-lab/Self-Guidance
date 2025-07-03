#!/bin/bash

accelerate launch inference/hps.py \
--config config/flux.yaml \
--local_data_path data/hpsv2 \
--output runs/flux/hps \
--seed 0 

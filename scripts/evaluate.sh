#!/bin/bash

python evaluation/evaluate.py \
--coco_data_path runs/flux/coco/cfg3.5-pag0-sg3_10_sg_500 \
--hps_data_path /storage/qiguojunLab/litiancheng/Self-Guidance/runs/flux/hps/cfg3.5-pag0-sg1_10_sg_500 \
--prompt_path data/coco/coco_val5000_prompts.txt \
--hps
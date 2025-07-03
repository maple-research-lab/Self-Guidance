# evaluate_scores.py

from fid_score import fid_coco
from aesthetic_score import aes_score
from clip_score import clip_score
from human_preference_score import hpsv

import argparse

def main(args):
    if args.fid:
        print("Calculating FID...")
        fid_coco(args.coco_data_path)

    if args.clip:
        print("Calculating CLIP score...")
        clip_score(args.coco_data_path, args.prompt_path)

    if args.aes:
        print("Calculating Aesthetic Score...")
        aes_score(args.coco_data_path)

    if args.hps:
        print("Calculating HPS score...")
        hpsv(args.hps_data_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--coco_data_path', type=str, required=True, help='Path to coco generated images')
    parser.add_argument('--hps_data_path', type=str, required=True, help='Path to hps generated images')
    parser.add_argument('--prompt_path', type=str, required=True, help='Path to coco prompts')
    parser.add_argument('--fid', action='store_true', help='Enable FID computation')
    parser.add_argument('--clip', action='store_true', help='Enable CLIP score computation')
    parser.add_argument('--aes', action='store_true', help='Enable Aesthetic score computation')
    parser.add_argument('--hps', action='store_true', help='Enable HPS computation')
    args = parser.parse_args()
    main(args)
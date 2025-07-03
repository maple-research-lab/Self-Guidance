from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

def fid_coco(path):
    fid, _ = calculate_fid(path,get_coco_fid_stats())
    return fid

def fid(generate_path, ground_truth_path):
    fid, _ = calculate_fid(generate_path,ground_truth_path)
    return fid



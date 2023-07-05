import os
import random
import shutil
import json
from distutils.log import debug
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from psql import connect_db, create_table, drop_table, insert_col, get_raw_output_paths
from types import *

class ShichimiDataset(Dataset):
    def __init__(self,
                 dataset_type: str,
                 path: str = "../data/"):

        print(f"====== {dataset_type} ======")

        data = self._build(path,dataset_type)
        self.data = data
        self.is_training = dataset_type == "train"

    def _build(self,path_str: str,dataset_type: str):
        with open(os.path.join(path_str, "stair_captions_v1.2_val.json"), 'r') as f:
            stair = json.load(f)

        images = stair["images"] # flickr_url
        annotations = stair["annotations"] # annotations
        image_to_imgid = {im["flickr_url"] : im["id"] for im in images}
        imgid_to_captions = {}
        for ann in annotations:
            imgid_to_captions.setdefault(ann["image_id"],[]).append(ann["caption"])
        
        with connect_db() as db:
            cursor = db.cursor()
            cursor.execute("SELECT im.id, c.caption, r.mos, im.url  FROM results r INNER JOIN captions c ON r.caption_id = c.id INNER JOIN images im ON im.id = c.image_id")
            data = cursor.fetchall()
            data = [(imgid,cap,mos,imgid_to_captions[image_to_imgid[url]]) for (imgid,cap,mos,url) in data]
            cursor.close()
            
        random.seed(42); random.shuffle(data)
        data = self._split(dataset_type,data)
        return data

    def _split(self,dataset_type,data):
        train_r, val_r, test_r = 0.8, 0.1, 0.1
        if dataset_type == "train":
            left = len(data) * 0
            right = len(data) * train_r
        elif dataset_type == "val":
            left = len(data) * train_r
            right = len(data) * (train_r + val_r)
        else:
            left = len(data) * (train_r + val_r)
            right = len(data) * (train_r + val_r + test_r)
        
        left, right = map(int,(left,right))
        return data[left:right]

    def __len__(self) -> int:
        """
        Returns:
            [int]: [length of sample]
        """
        return len(self.data)
    

    def __getitem__(self, idx: int):
        """
            get sample
        """
        img_id,cap,mos,gt_caps = self.data[idx]        
        return img_id, cap, gt_caps, mos


def main():
    splits = ["train", "val", "test"]
    datasets = [(s,ShichimiDataset(s)) for s in splits]
    # data,lp,src,ref,pos,neg,pos.model,neg.model,bestmodel


    # df = df[["src", "mt", "ref", "score"]]

    # ORIGINAL COMET
    # src: source
    # ref: gt
    # mt: hypo
    # score: DA

    # For image captioning
    # src: ref1
    # ref: ref2
    # mt: hypo
    # score: mos

    for split_name, dataset in datasets:
        print(f"original({split_name}): {len(dataset)}")
        img_to_tuples = {}
        for img_id, hypo, refs, mos in dataset:
            img_to_tuples.setdefault(img_id,[]).append((mos, hypo, refs))

        tuple_dataset = []
        for tpl in img_to_tuples.values():
            tpl.sort()
            for i, (crr_mos, crr_hypo, refs) in enumerate(tpl):
                N = len(refs)
                tuple_dataset.append({
                    "src": refs[i%N],
                    "ref": refs[(i+1)%N], 
                    "mt": crr_hypo,
                    "score": crr_mos,
                })
            
        print(f"new({split_name}): {len(tuple_dataset)}")
        with open(f"../data/shichimi_{split_name}_da.csv", "w") as f:
            f.write("src,mt,ref,score\n")
            for tpl in tuple_dataset:
                f.write(f"\"{tpl['src']}\",\"{tpl['mt']}\",\"{tpl['ref']}\",\"{tpl['score']}\"\n")

if __name__ == "__main__":
    main()
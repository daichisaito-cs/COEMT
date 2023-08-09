import json
# from jaspice.api import JaSPICE
from comet.metrics.regression_metrics import RegressionReport
from comet.models import load_checkpoint
import pandas as pd
import argparse
# from preprocess.psql import connect_db, create_table, drop_table, insert_col, get_raw_output_paths
from tqdm import tqdm
from os import path
from PIL import Image
import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from janome.tokenizer import Tokenizer
from jaspice.api import JaSPICE

def main():
    # print(args)
    dataset = pd.read_csv("data/shichimi_val_da.csv")
    print(dataset)
    imgids = dataset["imgid"]

    candidates = {i: [hypo] for i, hypo in enumerate(dataset["mt"])}
    # references = {i: imgid_to_captions[str(imgid)] for i, imgid in enumerate(dataset["imgid"])}
    references = {i: [ref, dataset["src"][i]] for i, ref in enumerate(dataset["ref"])}
    # print(references)
    gts = {i: mos for i, mos in enumerate(dataset["score"])}
    assert len(candidates) == len(references) == len(dataset)

    for imgid in candidates.keys():
        assert len(references[imgid]) == 2, len(references[imgid])

    def look_for_image(imgid, img_dir_path):
        img_name = path.join(img_dir_path, f"{imgid}.jpg")
        img = Image.open(img_name).convert("RGB")
        return img

    def is_image_ok(img_path):
        # Try to open the image file
        try:
            img = Image.open(img_path)
            img.verify()
            return True
        except (IOError, SyntaxError) as e:
            return False

    img_dir_path = "data/downloaded_images"
    # mycomet
    rep = RegressionReport()
    # model = load_checkpoint(args.model)
    # model = load_checkpoint("/home/initial/workspace/COMET/experiments/lightning/version_25-07-2023--17-21-55/epoch=2-step=1187.ckpt")
    model = load_checkpoint("/home/initial/workspace/COMET/experiments/lightning/version_25-07-2023--05-48-16/epoch=9-step=3959.ckpt")
    data = []
    data2 = []
    gt_scores = []
    flag = True
    for imgid, hypo in tqdm(candidates.items()):
        if flag:
            print(len(references[imgid]))
        flag = False
        if is_image_ok(f"{img_dir_path}/{imgids[imgid]}.jpg"):
            img_data = look_for_image(imgids[imgid], img_dir_path)
            data.append(
                {
                    "src": references[imgid][0],
                    "mt": hypo[0],
                    "ref": references[imgid][1],
                    "img": img_data,
                }
            )
            data2.append(
                {
                    "src": references[imgid][1],
                    "mt": hypo[0],
                    "ref": references[imgid][0],
                    "img": img_data,
                }
            )
            gt_scores.append(gts[imgid])
    # print(gt_scores)
    seg_scores, sys_score = model.predict(data,cuda=True)
    seg_scores, sys_score2 = model.predict(data2,cuda=True)

    # t = Tokenizer()
    # smoothie = SmoothingFunction().method2

    # def scaling(x):
    #     return -(x-1)*(x-1)+1

    # refs_sp = [[token.surface for token in t.tokenize(row["ref"])] for row in data]
    # mts_sp = [[token.surface for token in t.tokenize(row["mt"])] for row in data]
    # for j in range(len(data)):
    #     bleu_score = sentence_bleu([refs_sp[j]], mts_sp[j], smoothing_function=smoothie)
    #     # print(bleu_score)
    #     if sys_score[j] >= 0.1:
    #         sys_score[j] -= (1 - scaling(bleu_score)) / 10

    # refs_sp = [[token.surface for token in t.tokenize(row["ref"])] for row in data2]
    # mts_sp = [[token.surface for token in t.tokenize(row["mt"])] for row in data2]
    # for j in range(len(data)):
    #     bleu_score = sentence_bleu([refs_sp[j]], mts_sp[j], smoothing_function=smoothie)
    #     # print(bleu_score)
    #     if sys_score2[j] >= 0.1:
    #         sys_score2[j] -= (1 - scaling(bleu_score)) / 10
    max_values = []
    for i in range(len(sys_score)):
        max_values.append(max(sys_score[i], sys_score2[i]))
    metrics = rep.compute(max_values, gt_scores)

    print("COMET",metrics)

    # jaspice
    jaspice = JaSPICE(batch_size=16,server_mode=True)
    _, scores = jaspice.compute_score(references, candidates)
if __name__ == "__main__":
    main()

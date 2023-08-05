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

def main(args):
    print(args)
    dataset = pd.read_csv("data/pfnpic.csv")
    print(dataset)
    imgids = dataset["imgid"]
    imgid_to_captions = {}
    with open("data/pfnpic.json") as f:
        raws = json.load(f)
        for data in raws:
            imgid_to_captions[data["id"]] = data["references"]

    candidates = {i: [hypo] for i, hypo in enumerate(dataset["mt"])}
    references = {i: imgid_to_captions[str(imgid)] for i, imgid in enumerate(dataset["imgid"])}
    gts = {i: mos for i, mos in enumerate(dataset["score"])}
    # gt_scores = [gts[k] for k,_ in candidates.items()]
    assert len(candidates) == len(references) == len(dataset)

    for imgid in candidates.keys():
        # print(gts[imgid],candidates[imgid], references[imgid][0])
        assert len(references[imgid]) == 3, len(references[imgid])

    def look_for_image(imgid, img_dir_path):
        img_name = path.join(img_dir_path, f"{imgid}.png")
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


    img_dir_path = "data/pfnpic_images"
    # mycomet
    rep = RegressionReport()
    model = load_checkpoint(args.model)
    # "/home/initial/workspace/COMET/experiments/lightning/version_25-07-2023--16-39-52/epoch=1-step=791.ckpt"
    data = []
    gt_scores = []
    for imgid, hypo in tqdm(candidates.items()):
        if is_image_ok(f"{img_dir_path}/{imgids[imgid]}.png"):
            data.append(
                {
                    "src": references[imgid][0],
                    "mt": hypo[0],
                    "ref": references[imgid][1],
                    "img": look_for_image(imgids[imgid], img_dir_path),
                }
            )
            gt_scores.append(gts[imgid])

    seg_scores, sys_score = model.predict(data,cuda=True)
    metrics = rep.compute(sys_score, gt_scores)

    print("COMET",metrics)
    plt.figure()
    plt.hist(sys_score, bins='auto')
    plt.savefig("comet_score_samesize.png")

    # plt.figure()
    # plt.hist(gt_scores, bins='auto')
    # plt.savefig("gt_score.png")
   
    # jaspice
    # jaspice = JaSPICE(batch_size=16,server_mode=True)
    # _, scores = jaspice.compute_score(references, candidates)
    # for i, (k,v) in enumerate(list(candidates.items())[:20]):
    #     print(f"scores: {scores[i]} / gts: {gt_scores[i]}")
    #     print("hypo",v)
    #     print("gt",references[i],end="\n\n")

    # metrics = rep.compute(scores, gt_scores)
    # metrics = corrcoef(scores,gt_scores)
    # print("JaSPICE", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    main(args)

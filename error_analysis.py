import json
from jaspice.api import JaSPICE
from comet.metrics.regression_metrics import RegressionReport
from comet.models import load_checkpoint
import pandas as pd
import argparse
# from preprocess.psql import connect_db, create_table, drop_table, insert_col, get_raw_output_paths
from tqdm import tqdm
from os import path
from PIL import Image
import random
import os

def main(args):

    print(args)
    dataset = pd.read_csv("data/pfnpic.csv")

    # idxs = []
    # for i in range(10):
    #     idxs.append(random.randint(0, len(dataset)-1))
    # dataset = dataset.loc[idxs]
    # print("------references------")
    # print(dataset["mt"])
    # print("------imgid------")
    # print(dataset["imgid"])
    # print("------references------")
    # for i in dataset["references"]:
    #     print(i[0], i[1], i[2])
    # if not path.exists("data/error_analysis_images"):
    #     os.makedirs("data/error_analysis_images")
    # for i in dataset["imgid"]:
    #     img = Image.open(f"data/pfnpic_images/{i}.png")
    #     img.save(f"data/error_analysis_images/{i}.png")

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


    idxs = []
    random.seed(10)
    for i in range(10):
        idxs.append(random.randint(0, len(references)-1))
    # dataset = dataset.loc[idxs].reset_index(drop=True)
    imgids = imgids.loc[idxs].reset_index(drop=True)
    references = {i: references[key] for i, key in enumerate(idxs) if key in references}
    candidates = {i: candidates[key] for i, key in enumerate(idxs) if key in candidates}

    # Save images
    if not path.exists("data/error_analysis_images"):
        os.makedirs("data/error_analysis_images")
    for i in imgids:
        img = Image.open(f"data/pfnpic_images/{i}.png")
        img.save(f"data/error_analysis_images/{i}.png")


    gts = {i: gts[key] for i, key in enumerate(idxs)}


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
    rep = RegressionReport()
    # model = load_checkpoint(args.model)
    model = load_checkpoint("/home/initial/workspace/COMET/experiments/lightning/version_25-07-2023--17-21-55/epoch=2-step=1187.ckpt")

    img_status = {idx: is_image_ok(f"{img_dir_path}/{imgid}.png") for idx, imgid in enumerate(imgids.values)}
    img_data = {idx: look_for_image(imgid, img_dir_path) if status else None for idx, (imgid, status) in enumerate(zip(imgids.values, img_status.values()))}

    def create_data(src_idx, ref_idx):
        return [
            {
                "src": references[idx][src_idx],
                "mt": hypo[0],
                "ref": references[idx][ref_idx],
                "img": img_data[idx],
            }
            for idx, hypo in candidates.items() if img_status[idx]
        ]

    data_sets = [
        create_data(i, (i + 1) % 3) for i in range(3)
    ] + [
        create_data(i, (i + 2) % 3) for i in range(3)
    ]



    gt_scores = [gts[idx] for idx in range(len(candidates)) if img_status[idx]]

    sys_scores = [
        model.predict(data, cuda=True)[1] for data in tqdm(data_sets)
    ]

    metrics = rep.compute(sys_scores[0], gt_scores)

    max_values = [max(scores) for scores in zip(*sys_scores)]
    max_metrics = rep.compute(max_values, gt_scores)


    print(candidates)
    print(references)
    print(imgids)
    print(gt_scores)
    print(max_values)

    print("COMET", metrics)
    print("COMET-MAX", max_metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    main(args)

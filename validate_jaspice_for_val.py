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

def main():
    # print(args)
    dataset = pd.read_csv("data/shichimi_train_same_size.csv")
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
    model = load_checkpoint("/home/initial/workspace/COMET/experiments/lightning/version_03-08-2023--23-32-01/epoch=6-step=1686.ckpt")
    data = []
    gt_scores = []

    for imgid, hypo in tqdm(candidates.items()):
        if is_image_ok(f"{img_dir_path}/{imgids[imgid]}.jpg"):
            data.append(
                {
                    "src": references[imgid][0],
                    "mt": hypo[0],
                    "ref": references[imgid][1],
                    "img": look_for_image(imgids[imgid], img_dir_path),
                }
            )
            gt_scores.append(gts[imgid])
    # print(gt_scores)
    seg_scores, sys_score = model.predict(data,cuda=True)

    metrics = rep.compute(sys_score, gt_scores)

    # print("COMET",metrics)
    # plt.figure()
    # plt.hist(sys_score, bins='auto')
    # plt.savefig("comet_score_train.png")

    # plt.figure()
    # plt.hist(gt_scores, bins='auto')
    # plt.savefig("gt_score_train.png")

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--model')
#     args = parser.parse_args()
#     main(args)
if __name__ == "__main__":
    main()

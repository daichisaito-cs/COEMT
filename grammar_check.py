import json
from comet.metrics.regression_metrics import RegressionReport
from comet.models import load_checkpoint
import pandas as pd
import argparse
# from preprocess.psql import connect_db, create_table, drop_table, insert_col, get_raw_output_paths
from tqdm import tqdm
from os import path
from PIL import Image
import time
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def main():
    start_time = time.time()
    # print(args)
    dataset = pd.read_csv("data/pfnpic.csv")
    # print(dataset)
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

    t = Tokenizer()
    

    for i in data_sets:
        for j in i:
            bleu_score = 0.5
            sys_scores[i][j] += (bleu_score - 1) / 5

    metrics = rep.compute(sys_scores[0], gt_scores)

    max_values = [max(scores) for scores in zip(*sys_scores)]
    max_metrics = rep.compute(max_values, gt_scores)

    print("COMET", metrics)
    print("COMET-MAX", max_metrics)
    end_time = time.time()
    elapsed_time = end_time - start_time
    # print(elapsed_time)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main()

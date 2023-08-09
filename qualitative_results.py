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
    dataset = pd.read_csv("data/shichimi_val_da.csv")
    imgids = dataset["imgid"]
    with open("data/stair_captions_v1.2_val.json", 'r') as f:
        stair = json.load(f)

    images = stair["images"] # flickr_url
    annotations = stair["annotations"] # annotations
    image_to_imgid = {im["flickr_url"] : im["id"] for im in images}
    imgid_to_captions = {}
    for ann in annotations:
        imgid_to_captions.setdefault(ann["image_id"],[]).append(ann["caption"])

    candidates = {i: [hypo] for i, hypo in enumerate(dataset["mt"])}
    references = {i: imgid_to_captions[imgid] for i, imgid in enumerate(dataset["imgid"])}
    gts = {i: mos for i, mos in enumerate(dataset["score"])}
    # gt_scores = [gts[k] for k,_ in candidates.items()]
    assert len(candidates) == len(references) == len(dataset)

    for imgid in candidates.keys():
        # print(gts[imgid],candidates[imgid], references[imgid][0])
        assert len(references[imgid]) == 5, len(references[imgid])

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
    img_status = {idx: is_image_ok(f"{img_dir_path}/{imgid}.jpg") for idx, imgid in enumerate(imgids.values)}
    img_data = {idx: look_for_image(imgid, img_dir_path) if status else None for idx, (imgid, status) in enumerate(zip(imgids.values, img_status.values()))}

    gt_scores = [gts[idx] for idx in range(len(candidates)) if img_status[idx]]

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

    # data_sets = [
    #     create_data(i, (i + 1) % 5) for i in range(5)
    # ] + [
    #     create_data(i, (i + 2) % 5) for i in range(5)
    # ] + [
    #     create_data(i, (i + 3) % 5) for i in range(5)
    # ] + [
    #     create_data(i, (i + 4) % 5) for i in range(5)
    # ]

    data_sets = [create_data(0, 1)]

    sys_scores = [
        model.predict(data, cuda=True)[1] for data in tqdm(data_sets)
    ]

    def ngram_overlap(ref, mt, n=2):
        # Generate n-grams for the reference sentence
        ref_ngrams = set([tuple(ref[i:i+n]) for i in range(len(ref)-n+1)])
        
        # Generate n-grams for the machine translation sentence
        mt_ngrams = set([tuple(mt[i:i+n]) for i in range(len(mt)-n+1)])
        
        # Calculate the overlap
        overlap = ref_ngrams.intersection(mt_ngrams)
        
        # Return the overlap ratio
        return len(overlap) / len(ref_ngrams)
    
    t = Tokenizer()

    for i, data in enumerate(data_sets):
        refs_sp = [[token.surface for token in t.tokenize(row["ref"])] for row in data]
        mts_sp = [[token.surface for token in t.tokenize(row["mt"])] for row in data]
        for j in range(len(data)):
            # bleu_score = sentence_bleu([refs_sp[j]], mts_sp[j], smoothing_function=smoothie)
            bleu_score = ngram_overlap(refs_sp[j], mts_sp[j])
            if sys_scores[i][j] >= 0.1:
                if bleu_score <= 0.4:
                    sys_scores[i][j] -= (1 - bleu_score) / 10

    metrics = rep.compute(sys_scores[0], gt_scores)

    max_values = [max(scores) for scores in zip(*sys_scores)]
    max_metrics = rep.compute(max_values, gt_scores)

    sorted_results = []
    for i in range(len(max_values)):
        sorted_results.append({"gt": gt_scores[i], "comet": max_values[i], "mt":data_sets[0][i]["mt"], "ref": [data_sets[0][i]["src"], data_sets[0][i]["ref"]]})
    sorted_results = sorted(sorted_results, key=lambda x: abs(x["gt"] - x["comet"]))

    with open("qualitative_results.json", "w") as f:
        json.dump(sorted_results, f)

    print("COMET", metrics)
    print("COMET-MAX", max_metrics)
    # print(elapsed_time)
if __name__ == "__main__":
    main()

import json
from jaspice.api import JaSPICE
from comet.metrics.regression_metrics import RegressionReport
from comet.models import load_checkpoint
import pandas as pd
import argparse

def main(args):
    print(args)
    dataset = pd.read_csv("data/shichimi_val_da.csv")
    print(dataset)
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
    gt_scores = [gts[k] for k,_ in candidates.items()]
    assert len(candidates) == len(references) == len(dataset)

    for imgid in candidates.keys():
        assert len(references[imgid]) == 5


    # mycomet
    rep = RegressionReport()
    model = load_checkpoint(args.model)
    data = [
        {
            "src": references[imgid][0],
            "mt": hypo[0],
            "ref": references[imgid][1],
        }
        for imgid, hypo in candidates.items()
    ]
    seg_scores, sys_score = model.predict(data,cuda=True)
    metrics = rep.compute(sys_score, gt_scores)
    # metrics = corrcoef(sys_score, gt_scores)

    print("COMET",metrics)

    # jaspice
    jaspice = JaSPICE(batch_size=16,server_mode=True)
    _, scores = jaspice.compute_score(references, candidates)
    # for i, (k,v) in enumerate(list(candidates.items())[:20]):
    #     print(f"scores: {scores[i]} / gts: {gt_scores[i]}")
    #     print("hypo",v) 
    #     print("gt",imgid_to_captions[k],end="\n\n")

    metrics = rep.compute(scores, gt_scores)
    # metrics = corrcoef(scores,gt_scores)
    print("JaSPICE", metrics)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model')
    args = parser.parse_args()
    main(args)

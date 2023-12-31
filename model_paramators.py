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
from torchinfo import summary
import torch

def main():
    dataset = pd.read_csv("data/shichimi_train_same_size.csv")
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
    model = load_checkpoint("/home/initial/workspace/COMET/experiments/lightning/version_03-08-2023--07-46-58/epoch=3-step=963.ckpt")
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
    src_tokens = torch.tensor([[     0,   5140,     14,    251, 125954,   4130,  15957,    281,    304,
            487,   8872,      2,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 195299,    281,   3195,  69188,  16974,  39564,   3014,
           3439,  61212,  55524,      2,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,   3515,   2111, 104261,    154,  63167,  63763,   2458,
            327,   2112,  10930,    281,   2427,  12087,      2,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 187488,    327,   3439,  28520,  76802,  81832,   3385,
            251, 114925, 179040,   4020,   7229,   8421,   7826,      2,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  73619,   2026,    507,  17462,   5677,   3385,  84693,
          29040,  83445,    610,  37084,      2,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  76802,   3515,  42973,    610, 139142,    281, 159081,
          43422,  19664,    327,  17742,  67955,  15065,      2,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 233379,  10045,  97659,  17852,    154, 108913,  36979,
           2111,  20629,  55658,  13523,   7794,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,   3515,   2111, 238030,    327,  17462,  55658,  14760,
         161299,  17444,  99711,    154,  12087,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  68770,   7116,  17628,  76872,    154, 217618,    327,
           3385,  92954,    154,   2112,  10930,    281,   9449,  12087,      2,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,   9368,   9368,  17456, 219124,  17628,   5283,   4758,
          11804,  25279, 158676,  37959,  21300,    610,  78694, 159341,    610,
          15343,  11119,  13451,   9368,  21711,    281,   2636, 238030,  24433,
              2,      1],
        [     0,      6,  10134, 201187,   6375,    154,  49969,   9421,  12087,
          41473,    610,  51241,  46713,  61212,  12087,  41473,      2,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  21975,    281,   2283,   6219,  32607,   4379,    154,
           5032,    281,   4121,   7229,  10449,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  14323,  67540,    281,  22773,  24961,    251,  16698,
          64870,   4020,   7229,   8421,   7826,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  36979,   4379,    154, 171688,  93250,    251,   1562,
           2427,  72658,    610, 183723,   4379,    154, 171688,  93250,    251,
           1562,   2427,  72658,    281, 204257,    154, 108517,  58497,      2,
              1,      1],
        [     0,      6, 209798,  18248,    154,  21200,    281,   8924,    610,
         157936,    327,   2825,  12087,      2,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 183723,    154,   2026,    955,    154,   2603, 100315,
            154, 217618,  58497,  27857, 105373, 195299,   7794,      2,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  65322, 163083,    154,  25838,  82520,    154, 199891,
            327,   6656,  58256,  31647, 104261,   7794,      2,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 210420, 217192,    507,  92356,    251,  35053,   1894,
           4130,   8028, 119269,      2,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 123159,    327,  13082,   6219, 173152,   3014,  31477,
         233379,   8929,      2,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  12802,  10970,    154,  40806,    251, 182160,    327,
          55642,   6219,  19758,   3469,  12087,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 167776, 210420, 217192,    507,  92356,    251,  35053,
           1894,   4130,      2,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 207280,  78609,    327,  74496,   2111,  35076,    281,
          40806,  45063,      2,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 212902,    154,  10930,  45957, 215495,    487, 119269,
              2,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,  70436,    605,  79909,    610,   4525, 123096, 102209,   3752,
            281, 102209,   4130,      2,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  12819,   5283,  37959,    154,  31477,  50866,    251,
          12802,  59368,      2,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 189596,  45651,  20629,  55658,  13523,    281,    363,
           3275,   3439,  61212,  55524,      2,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,   1553, 114026,    327,    363,   4138,    154,  24082,
            281,  14031,     37,   3230,  17206,   2880, 183723,   4379,    154,
           2026,    955,    507,  10351,   2427,    635, 156303,   9586,  13871,
           7794,      2],
        [     0,  87744,   3752,    154,  26221,   3385,  17456,    281,   8924,
           2111,   5841,    251,  21062,  59368,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,   1553,   4379,    610,  14577,  55658,  13523,    327,
           4379, 139380,   8346, 148156,    154,   4411,   8481,    281, 116209,
           2694,   3985,  12087,      2,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 127363, 214092,    327,  23279,   4758,  84682,  82038,
          15957,    281, 210413,  16985,  80137,      2,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6,  35076,    281,  48418,    327,  69223,   2075,   2825,
            327, 136166,   4130,      2,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1],
        [     0,      6, 167776,  18889, 114543,   3385, 176227,  13523, 119790,
              2,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1,      1,      1,      1,      1,      1,      1,      1,
              1,      1]], device='cuda:0')
    mt_tokens = src_tokens
    ref_tokens = src_tokens
    src_lengths= torch.tensor([12, 13, 16, 17, 14, 16, 15, 15, 18, 28, 17, 15, 15, 27, 14, 17, 16, 13,
        12, 15, 12, 12, 10, 13, 12, 14, 29, 15, 22, 15, 13, 10],
       device='cuda:0')
    mt_lengths = src_lengths
    ref_lengths = src_lengths
    src_idf =  torch.tensor([[0.0000, 7.0568, 5.8067, 1.1217, 7.6259, 2.4466, 2.0233, 0.5051, 3.3928,
         3.5244, 2.3911, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.5579, 0.5051, 4.0927, 6.0165, 4.9179, 5.0713, 4.0141,
         3.2278, 4.1519, 3.2586, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 2.4396, 1.4555, 4.2247, 0.5784, 4.6572, 5.0817, 4.1770,
         1.2146, 4.5688, 4.2006, 0.5051, 2.3708, 1.5523, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.6259, 1.2146, 3.2278, 3.8835, 3.6222, 4.4191, 2.1772,
         1.1217, 4.6136, 5.0282, 5.9645, 3.5249, 4.4030, 2.4481, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.8225, 3.9420, 1.9377, 3.5478, 3.2895, 2.1772, 4.2925,
         5.0922, 5.7288, 1.9151, 3.4759, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.6222, 2.4396, 3.8097, 1.9151, 7.8082, 0.5051, 5.4819,
         4.1570, 5.6066, 1.2146, 4.2529, 6.1293, 4.2027, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.5273, 4.1560, 6.5963, 4.3141, 0.5784, 2.5264, 2.6231,
         1.4555, 5.9772, 3.3736, 3.3409, 2.8468, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 2.4396, 1.4555, 3.4302, 1.2146, 3.5478, 3.3736, 3.8104,
         5.5761, 5.0134, 5.3923, 0.5784, 1.5523, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 6.3924, 3.1975, 4.2192, 5.2525, 0.5784, 6.0998, 1.2146,
         2.1772, 5.1692, 0.5784, 4.5688, 4.2006, 0.5051, 4.9381, 1.5523, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.9320, 3.9320, 3.4044, 5.3711, 4.2192, 3.7499, 3.4025,
         6.0032, 5.1355, 4.9313, 4.5154, 4.4044, 1.9151, 6.3924, 5.3035, 1.9151,
         3.7736, 4.1602, 3.4148, 3.9320, 4.0222, 0.5051, 4.9797, 3.4302, 4.4084,
         0.0000, 0.0000],
        [0.0000, 0.0000, 6.2480, 5.6705, 3.8297, 0.5784, 4.7172, 3.0962, 1.5523,
         4.7227, 1.9151, 4.4479, 5.2938, 4.1519, 1.5523, 4.7227, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.6956, 0.5051, 4.4164, 2.8580, 6.3176, 2.6950, 0.5784,
         4.2426, 0.5051, 5.2371, 3.5249, 2.9600, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.6468, 3.6753, 0.5051, 6.1218, 5.0765, 1.1217, 4.5974,
         6.3450, 5.9645, 3.5249, 4.4030, 2.4481, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 2.6231, 2.6950, 0.5784, 4.4732, 5.3101, 1.1217, 3.0039,
         2.3708, 6.5385, 1.9151, 4.0986, 2.6950, 0.5784, 4.4732, 5.3101, 1.1217,
         3.0039, 2.3708, 6.5385, 0.5051, 5.0687, 0.5784, 4.7947, 2.8235, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.6419, 5.3887, 0.5784, 6.2068, 0.5051, 3.1770, 1.9151,
         3.8670, 1.2146, 5.1002, 1.5523, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.0986, 0.5784, 3.9420, 4.6104, 0.5784, 4.0461, 5.2220,
         0.5784, 6.0998, 2.8235, 5.3134, 5.2557, 4.5579, 2.8468, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 6.5963, 6.2480, 0.5784, 4.3400, 4.3601, 0.5784, 4.9964,
         1.2146, 6.2068, 6.7230, 5.3035, 4.2247, 2.8468, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 5.0765, 5.1578, 1.9377, 4.2325, 1.1217, 5.2371, 2.7671,
         2.4466, 3.8410, 3.2200, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 5.6992, 1.2146, 7.0568, 2.8580, 7.3382, 4.0141, 3.5523,
         4.5273, 3.9555, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 5.9709, 7.2204, 0.5784, 3.4754, 1.1217, 3.8819, 1.2146,
         3.3332, 2.8580, 3.3187, 3.6035, 1.5523, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.1482, 5.0765, 5.1578, 1.9377, 4.2325, 1.1217, 5.2371,
         2.7671, 2.4466, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.6641, 3.7212, 1.2146, 2.7081, 1.4555, 3.3196, 0.5051,
         3.4754, 5.2011, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 4.0583, 0.5784, 4.2006, 4.8286, 7.0759, 3.5244, 3.2200,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 8.1939, 8.1939, 8.1939, 1.9151, 5.6800, 4.4164, 4.3425, 4.1124,
         0.5051, 4.3425, 2.4466, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.9752, 3.7499, 4.5154, 0.5784, 3.5523, 6.3450, 1.1217,
         5.9709, 3.1782, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 6.8527, 6.0926, 5.9772, 3.3736, 3.3409, 0.5051, 4.7888,
         3.1880, 3.2278, 4.1519, 3.2586, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.5178, 4.4562, 1.2146, 4.7888, 3.4875, 0.5784, 4.6285,
         0.5051, 4.8246, 2.4897, 4.7752, 4.8701, 3.6889, 4.0986, 2.6950, 0.5784,
         3.9420, 4.6104, 1.9377, 6.5612, 2.3708, 5.8566, 6.0299, 7.7313, 4.5349,
         2.8468, 0.0000],
        [0.0000, 5.1865, 4.1124, 0.5784, 5.7238, 2.1772, 3.4044, 0.5051, 3.1770,
         1.4555, 4.2854, 1.1217, 4.4437, 3.1782, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.5178, 2.6950, 1.9151, 3.4801, 3.3736, 3.3409, 1.2146,
         2.6950, 5.6289, 4.3678, 3.6556, 0.5784, 5.8738, 5.7801, 0.5051, 4.9587,
         3.4966, 5.3003, 1.5523, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 7.6949, 4.1054, 1.2146, 4.1345, 3.4025, 4.2621, 4.6202,
         2.0233, 0.5051, 4.3301, 3.5626, 7.0568, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.3196, 0.5051, 4.7227, 1.2146, 5.3746, 3.1497, 5.1002,
         1.2146, 7.7690, 2.4466, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000],
        [0.0000, 0.0000, 3.1482, 5.4213, 4.7986, 2.1772, 8.4622, 3.3409, 4.4070,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.0000, 0.0000]], device='cuda:0')
    mt_idf = src_idf
    ref_idf = src_idf
    src = ['Wiiをプレイしている男性が2人いる', 'ベッドが二つくっつけて並べられている', '白い建物の前の歩道に看板がたっている', 'テーブルに並ぶたくさんのグラスを小さな子供が見つめている', '芝生でフリスビーをする親子と犬', 'たくさんの白鳥と鴨が泳ぐ湖に停まる船', 'キッチンカウンターの上に赤い電子レンジがある', '白い皿にフレンチトーストなどがのっている', 'サイドミラーのカバーにストップの看板が写っている', 'ココナッツミルクのかかったハンバーグとピータンとブロッコリーがお皿にある', '組まれた木の上で座っている熊と寝そべっている熊', '料理が入った銀色の器が四つある', 'キリンが右側をじっと見つめている', '赤色のユニホームを着たチームと緑色のユニホームを着たチームが野球の試合をしている', '時計塔の文字が青と黄色に光っている', '緑の生地の花柄のカバーをしているダブルベッドがある', '一方通行の標識の後ろに落書きのある建物がある', 'ドライヤーで髪を乾かしている女性がいる', '綺麗に整った広くて大きなキッチンである', '運河の横を自転車に乗った人が走っている', '女性がドライヤーで髪を乾かしている', 'ノートパソコンに黒い猫が横になっている', 'スキーの板を持って歩く人がいる', 'TURKISHと書かれた飛行機が飛行している', 'シルバーの大きな機械を運んでいる', 'タイプの違う電子レンジが3台並べられている', '水辺に3頭の象がいて、その奥には緑色の生地で出来た日差し除けがある', '二機のセスナが青い空を飛んでいる', '水色とオレンジに色分けされたトイレの個室が隣り合っている', 'ワイシャツにネクタイをした男性がジャンプしてる', '猫がテレビに映る光に反応している', '女性が左手にスポンジを持っている']
    imgs = [torch.randn(32, 1, 480, 640)]
    alt_tokens = src_tokens
    alt_lengths = src_lengths,
    print(summary(model, (src_tokens, mt_tokens, ref_tokens, src_lengths, mt_lengths, ref_lengths, src_idf, mt_idf, ref_idf, src, imgs, alt_tokens, alt_lengths)))
if __name__ == "__main__":
    main()

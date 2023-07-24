# -*- coding: utf-8 -*-
r"""
Comet Estimator Model
================================
    Comet Estimator predicts a quality score for the
    hyphotesis (e.g: the MT text) by looking at reference, source and MT.
"""
import random
from argparse import Namespace
from typing import Dict, List, Tuple, Union

import torch
# from torchvision.transforms import ToTensor

from comet.models.estimators.estimator_base import Estimator
from comet.modules.feedforward import FeedForward
from comet.modules.scalar_mix import ScalarMixWithDropout
from torchnlp.utils import collate_tensors
import numpy as np
import torch
import cv2

from typing import List, Union
from skimage import measure

import numpy as np

try:
    # ignore ShapelyDeprecationWarning from fvcore
    import warnings

    from shapely.errors import ShapelyDeprecationWarning

    warnings.filterwarnings("ignore", category=ShapelyDeprecationWarning)
except:
    pass
import os

import huggingface_hub
import torch
# from detectron2.checkpoint import DetectionCheckpointer
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2.engine import DefaultTrainer
# from detectron2.projects.deeplab import add_deeplab_config
# from detectron2.utils.visualizer import Visualizer, random_color
from huggingface_hub import hf_hub_download
from PIL import Image

# from san import add_san_config
# from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES
from comet.models.utils import average_pooling, max_pooling, move_to_cpu, move_to_cuda


# MAX_SEG_LABEL = 200 # 必ず写っている物体数には限りがあるので200にしておく
# VISUALIZE = False

# BACKBONE = "san_vit_b_16"
# model_cfg = {
#     "san_vit_b_16": {
#         "config_file": "configs/san_clip_vit_res4_coco.yaml",
#         "model_path": "huggingface:san_vit_b_16.pth",
#     },
#     "san_vit_large_16": {
#         "config_file": "configs/san_clip_vit_large_res4_coco.yaml",
#         "model_path": "huggingface:san_vit_large_14.pth",
#     },
# }


class TransformerEncoder(torch.nn.Module):
    def __init__(self, input_dim, num_heads, hidden_dim, num_layers, dropout=0.5):
        super(TransformerEncoder, self).__init__()
        self.input_dim = input_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout

        self.positional_encoding = torch.nn.Embedding(1000, input_dim)
        self.encoder_layer = torch.nn.TransformerEncoderLayer(d_model=input_dim,
                                                        nhead=num_heads,
                                                        dim_feedforward=hidden_dim,
                                                        dropout=dropout)
        self.transformer_encoder = torch.nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def forward(self, src, src_key_padding_mask):
        src = src + self.positional_encoding(torch.arange(src.size(0), device=src.device)).unsqueeze(1)
        output = self.transformer_encoder(src,src_key_padding_mask=src_key_padding_mask.transpose(0,1)) # TODO: チェック
        return output


def download_model(model_path: str):
    """
    Download the model from huggingface hub.
    Args:
        model_path (str): the model path
    Returns:
        str: the downloaded model path
    """
    if "HF_TOKEN" in os.environ:
        huggingface_hub.login(token=os.environ["HF_TOKEN"])
    model_path = model_path.split(":")[1]
    model_path = hf_hub_download("Mendel192/san", filename=model_path)
    return model_path


# def setup(device=None):
#     """
#     Create configs and perform basic setups.
#     """
#     cfg = get_cfg()
#     # for poly lr schedule
#     add_deeplab_config(cfg)
#     add_san_config(cfg)
#     cfg.merge_from_file(model_cfg[BACKBONE]["config_file"])
#     cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
#     cfg.freeze()
#     return cfg

class CometEstimator(Estimator):
    """
    Estimator class that uses a pretrained encoder to extract features from
    the sequences and then passes those features to a feed forward estimator.

    :param hparams: Namespace containing the hyperparameters.
    """

    class ModelConfig(Estimator.ModelConfig):
        switch_prob: float = 0.0

    def __init__(
        self,
        hparams: Namespace,
    ) -> None:
        super().__init__(hparams)

    def _build_model(self) -> Estimator:
        """
        Initializes the estimator architecture.
        """
        super()._build_model()

        vocabulary = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        # vocabulary.extend(['art', 'pentagon', 'colors', 'mustard', 'pills', 'oraqng', 'candy', 'sponch', 'circles', 'bear', 'crayons', 'lines', 'yellow', 'teperature', 'wrap', 'strap', 'coco', 'cement', 'mug', 'hexagon', 'bottom', 'chips', 'te', 'stand', 'please', 'bulb', 'sleeper', 'squirt', 'wit', 'obtain', 'hexagonal', 'stripe', 'erasers', 'diagnoal', 'shape', 'pink', 'it', 'cube', 'papers', 'boxq', 'coke', 'bands', 'with', 'measuring', 'wash', 'thread', 'pakage', 'soda', 'beige', 'place', 'vitamins', 'whie', 'structure', 'below', 'right-hand', 'photo', 'silper', 'gardasil', 'chappel', 'room', 'bootle', 'half', 'toothpaste', 'numbers', 'cap', 'disk', 'thermameter', 'product', 'soap', 'word', 'color', 'card', 'mouse', 'packages', 'leather', 'rectangular', 'animal', 'water', 'powder', 'plushie', 'case', 'layer', 'shope', 'balls', 'stripes', 'wrapped', 'disc', 'smaill', 'push', 'jane', 'expo\\', 'thermos', 'mall', '[', 'dull', 'manilla', 'container', 'ju', 'teddie', 'mivecthe', 'v-band', 'wholes', 'bano', 'tupe', 'kids\\', 'tonthe', 'octagon', 'pocket', 'cans', 'squishy', 'lrft', 'stubby', 'markings', 'scripture', 'spanch', 'calculator', 'bottles', 'staples', 'ot', 'ont', 'shield', 'k', 'cotton', 'jar', 'woman', 'child', 'eraser', 'ribbon', 'coaster', 'booklet', 'bx', 'rectangle', 'yello', 'lower', 'mark', 'board', 'holder', 'trowel', 'location', 'lid', 'bunch', 'cat', 'scale', 'sui', 'closest', 'space', 'star', 'botton', 'outline', 'orange', 'bux', 'box', 'thermometer', 'inside', 'glows', 'end', 'punchcard', 'bowl', 'fallen', 'plushy', 'gatsby', 'from', 'https', 'mixer', 'flag', 'cone', 'bottommost', 'chocolate', 'marks', 'caps', 'gren', 'ju-band', 'compartment', 'sticks', 'spongel', 'shapes', 'pick', 'logo', 'felt', 'bow', 'dispenser', 'burgondy', 'tape', 'snacks', 'lettering', 'bun', 'code', 'food', 'bean', 'palce', 'squeeze', 'mono', 'visa', 'spots', 'outer', 'sponge', 'writings', 'man', 'slippers', 'tea', 'brown', 'notepad', 'cardboard', 'torch', 'bottle', 'wheel', 'face', 'mercury', 'wooden', 'box-', 'gove', 'flipflop', 'cereal', 'gold', 'tl', 'int', 'symbol', 'cellaphane', 'bag', 'item', 'cd', 'bar', 'pale', 'rigth', 'pig', 'spach', 'pins', 'polythene', 'figurine', 'right', 'flipflops', 'sketch', 'flops', 'squeezy', 'drop', 'soup', 'mvoe', 'catalogue', 'dvd', 'darker', 'sandals', 'work', 'keychain', 'package', 'pad', 'eh', 'wrist', 'towards', 'words', 'tins', 'flip', 'packaged', 'triangular', 'corners', 'upright', 'pumpkin', 'teal', 'characters', 'needs', 'bags', 'bars', 'cartridge', 'meter', 'position', 'drawing', 'heart', 'tube', 'footwear', 'metal', 'into', 'kids', 'lefft', 'cassette', 'arms', 'recipe', 'cake', 'patten', 'stick', 'ball', 'pompoms', 'rgiht', 'krayons', 'flat', 'shoe', 'envelope', 'bottem', 'pencil', 'sauce', 'lable', 'barcode', 'cock', 'sandel', 'collar', 'above', 'flecks', 'pencils', 'spray', 'body', 'gel', 'pear', 'cream', 'on', 'items', 'handel', 'flop', 'thing', 'silver', 'circle', 'poms', 'tab', '\\', 'shiny', 'qtips', 'image', 'glue', 'temperature', 'light', 'cylinder', 'aid', 'rear', 'note', 'collection', 'get', 'yelow', 'thank', 'foil', 'grey', 'packed', 'lotion', 'cup', 'people', 'date', 'bandaid', 'chop', 'put', 'oil', 'insert', 'o', 'gems', 'stickers', 'inscription', 'colored', 'movie', 'band-aids', 'kleenex', 'brand', 'horizon', 'border', 'crayon', 'heard', 'half-black', 'packet', 'side', 'trim', 'bo', 'botom', 'pair', 'stack', 'wood', 'label', 'crackers', 'ha', 'clips', 'tan', 'shinny', 'tissues', 'sclipper', 'center', 'plastic', 'blower', 'spancha', 'colour', 'lime', 'hand', 'move', 'expo', 'coliur', 'drawer', 'aqua', 'tin', 'sticker', 'pen', 'teadybear', 'tot', 'spongy', 'edges', 'letter', 'store', 'containers', 'strip', 'condiment', 'cesar', 'grey/silver', 'drink', 'diagonal', 'toy', 'joke', 'loud', 'handle', 'mix', 'figure', 'rubber', 'remove', 'cork', 'stapler', 'pens', 'pattern', 'back', 'pickup', 'aids', 'packer', 'stuffed', 'pill', 'adjacent', 'cubicle', 'printing', 'katsup', 'leftmost', 'sports', 'medium', 'key', 'corner', 'one', 'video', 'cellophane', 'cubby', 'post', 'edge', 'lunch', ']', 'markers', 'folgers', 'tag', 'rope', 'text', 'men', 'pom', 'women', 'picture', 'medicine', 'cocacola', 'foot', 'egg', 'celsius', 'out', 'garden', 'size', 'seal', 'verticle', 'middle', 'thermometre', 'bin', 'coffee', 'chepal', 'vitamin', 'block', 'mugs', 'klunex', 'half-orange', 'pouch', 'marbles', 'left-hand', 'pin', 'design', 'translucent', 'band-aid', 'showing', 'book', 'gloves', 'peach', 'packing', 'candies', 'chappal', 'animals', 'writing', 'band', 'paste', 'jokes', 'teddybear', 'ticket', 'name', 'nozzle', 'packs', 'transparent', 'drag', 'laugh', 'mini', 'chain', 'topped', 'quadrant', 'labels', 'object', 'hold', 'colur', 'jel', 'kangaroo', 'letters', 'section', 'maroon', 'sky', 'plain', 'cover', 'portion', 'compact', 'top', 'person', 'cola', 'diagnol', 'stamps', 'cabinet', 'whiter', 'angle', 'form', 'upper', 'caste', 'pot', 'blak', 'clear', 'th', 'triangle', 'dark', 'apple', 'sheet', 'pop', 'grab', 'loer', 'print', 'line', 'type', 'and', 'balloon', 'wall', 'pick-up', 'thelower', 'apartment', '..', 'foam', 'circular', 'lot', 'appartment', 'black', 'green', 'flaps', 'bellow', 'close', 'copper', 'wristband', 'coca', 'chip', 'wrapping', 'sachet', 'plush', 'pack', 'bandaids', 'tissue', 'farthest', 'key-chain', 'triangles', 'dog', 'background', 'baggie', 'canister', 'aluminum', 'coloue', 'lemon', 'sandal', 'glove', 'buds', 'movies', 'in', 'objects', 'tanslucent', 'square', 'area', 'tibthe', 'situate', 'envelopes', 'dots', 'piggy', 'tootpaste', 'metallic', 'theupper', 'juband', 'capsules', 'noodles', 'cylindrical', 'sponce', 'stabler', 'sock', 'love', 'the', 'number', 'ash', 'doll', 'smiley', 'dish', 'slipper', 'photograph', 'multi', 'bears', 'teddy', 'bar-code', 'drinks', 'tray', 'blastic', 'toys', 'boy', 'flip-flop', 'packaging', 'purple', 'straps', 'humans', 'cokes', 'front', 'catch', 'couple', 'sliver', 'bottleand', 'bandages', 'blond', 'next', 'left', 'part', 'rightmost', 'beads', 'sqaure', 'coca-cola', 'hox', 'pastel', 'hearts', 'to', 'deposit', 'boxes', 'roll', 'movedt', 'duster', 'thats', 'books', 'alphabets', 'up', 'tip', 'rectangles', 'persons', 'dot', 'clip', 'oval', 'right-most', 'battery', 'chat', 'multicolour', 'chopsticks', 'holes', 'round', 'chest', 'ketchup', 'tubes', 'let', 'receipt', 'blue', 'paper', 'thong', 'eyre', 'palish', 'material', 'rack', 'button', 'cards'])
        vocabulary = list(set([v.lower().strip() for v in vocabulary]))

        if self.hparams.encoder_model != "LASER":
            self.layer = (
                int(self.hparams.layer)
                if self.hparams.layer != "mix"
                else self.hparams.layer
            )

            self.scalar_mix = (
                ScalarMixWithDropout(
                    mixture_size=self.encoder.num_layers,
                    dropout=self.hparams.scalar_mix_dropout,
                    do_layer_norm=True,
                )
                if self.layer == "mix" and self.hparams.pool != "default"
                else None
            )

        input_emb_sz = (
            self.encoder.output_units * 7
            if self.hparams.pool != "cls+avg"
            else self.encoder.output_units * 2 * 7
        )

        self.ff = torch.nn.Sequential(*[
            FeedForward(
                in_dim=input_emb_sz,
                hidden_sizes=self.hparams.hidden_sizes,
                activations=self.hparams.activations,
                dropout=self.hparams.dropout,
                final_activation=(
                    self.hparams.final_activation
                    if hasattr(
                        self.hparams, "final_activation"
                    )  # compatability with older checkpoints!
                    else "Sigmoid"
                ),
            ),
            torch.nn.Sigmoid()
        ])

        # self.transformer = TransformerEncoder(input_dim=768, num_heads=8, hidden_dim=768, num_layers=3)

        # model_path = model_cfg[BACKBONE]["model_path"]
        # cfg = setup()
        # self.san = DefaultTrainer.build_model(cfg)
        # if model_path.startswith("huggingface:"):
        #     model_path = download_model(model_path)
        # print("Loading model from: ", model_path)
        # DetectionCheckpointer(self.san, save_dir=cfg.OUTPUT_DIR).resume_or_load(model_path)
        # print("Loaded model from: ", model_path)

        # self.vocabulary = self._merge_vocabulary(vocabulary)



    def configure_optimizers(
        self,
    ) -> Tuple[List[torch.optim.Optimizer], List[torch.optim.lr_scheduler.LambdaLR]]:
        """ Sets different Learning rates for different parameter groups. """
        layer_parameters = self.encoder.layerwise_lr(
            self.hparams.encoder_learning_rate, self.hparams.layerwise_decay
        )
        ff_parameters = [
            {"params": self.ff.parameters(), "lr": self.hparams.learning_rate}
        ]

        if self.hparams.encoder_model != "LASER" and self.scalar_mix:
            scalar_mix_parameters = [
                {
                    "params": self.scalar_mix.parameters(),
                    "lr": self.hparams.learning_rate,
                }
            ]

            optimizer = self._build_optimizer(
                layer_parameters + ff_parameters + scalar_mix_parameters
            )
        else:
            optimizer = self._build_optimizer(layer_parameters + ff_parameters)
        scheduler = self._build_scheduler(optimizer)
        return [optimizer], [scheduler]

    def prepare_sample(
        self, sample: List[Dict[str, Union[str, float]]], inference: bool = False
    ) -> Union[
        Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]], Dict[str, torch.Tensor]
    ]:
        """
        Function that prepares a sample to input the model.

        :param sample: list of dictionaries.
        :param inference: If set to true prepares only the model inputs.

        :returns: Tuple with 2 dictionaries (model inputs and targets).
            If `inference=True` returns only the model inputs.
        """
        sample = collate_tensors(sample)
        src_inputs = self.encoder.prepare_sample(sample["src"])
        mt_inputs = self.encoder.prepare_sample(sample["mt"])
        ref_inputs = self.encoder.prepare_sample(sample["ref"])

        src_inputs = {"src_" + k: v for k, v in src_inputs.items()}
        mt_inputs = {"mt_" + k: v for k, v in mt_inputs.items()}
        ref_inputs = {"ref_" + k: v for k, v in ref_inputs.items()}

        if "alt" in sample:
            alt_inputs = self.encoder.prepare_sample(sample["alt"])
            alt_inputs = {"alt_" + k: v for k, v in alt_inputs.items()}
            inputs = {**src_inputs, **mt_inputs, **ref_inputs, **alt_inputs}

        else:
            inputs = {**src_inputs, **mt_inputs, **ref_inputs}

        get_idf = sample["idf_fn"][0]
        inputs["imgs"] = sample["img"]
        inputs["src"] = sample["src"]
        inputs["src_idf"] = get_idf(src_inputs["src_tokens"])
        inputs["mt_idf"] = get_idf(mt_inputs["mt_tokens"])
        inputs["ref_idf"] = get_idf(ref_inputs["ref_tokens"])

        # print(inputs["src"])
        one_sample = self.encoder.tokenizer.batch_decode(src_inputs["src_tokens"],src_inputs["src_lengths"])[0]
        one_sample_idf = inputs["src_idf"][0]
        print(one_sample)
        print(one_sample_idf)

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    # def patchify(self, img, patch_size=32):
    #     assert len(img.shape) == 4, "4D tensors expected"
    #     b, c, h, w = img.shape
    #     assert w % patch_size == 0 and h % patch_size == 0

    #     unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    #     patches = unfold(img)
    #     patches = patches.permute(0, 2, 1) # b, c, n
    #     return patches

    # def _merge_vocabulary(self, vocabulary: List[str]) -> List[str]:
    #     default_voc = [c["name"] for c in COCO_CATEGORIES]
    #     return vocabulary + [c for c in default_voc if c not in vocabulary]

    # def calculate_positional_encoding(self, height, width, dim_model,device):
    #     # Positional encoding for 2D images
    #     pe = torch.zeros(height, width, dim_model)
    #     y_position = torch.arange(0, height).unsqueeze(1)
    #     x_position = torch.arange(0, width).unsqueeze(1)
    #     div_term = torch.exp(torch.arange(0., dim_model, 2) * -(np.log(10000.0) / dim_model))

    #     pe[:, :, 0::2] = torch.sin(y_position * div_term)
    #     pe[:, :, 1::2] = torch.cos(x_position * div_term)
    #     return pe.to(device)

    # def create_embeddings_from_mask(self, maskes, num_labels, label_emb):
    #     from cc_torch import connected_components_labeling
    #     B, H, W = maskes.shape
    #     batch_embeddings = []

    #     # create positional encoding
    #     positional_encoding = self.calculate_positional_encoding(H, W, label_emb.shape[-1], device=label_emb.device)

    #     # vocab = self._merge_vocabulary(self.vocabulary)
    #     for b in range(B):
    #         mask = maskes[b,:,:]
    #         # label each connected component with a unique id
    #         labelled_mask = connected_components_labeling(mask.to(torch.uint8)) # TODO: uint8でいいんだっけ...?????
    #         embeddings = []
    #         for i in range(num_labels):
    #             indices = (labelled_mask == i)
    #             if indices.any():
    #                 label = mask[indices][0]
    #                 emb = label_emb[label] + positional_encoding[indices].mean(dim=0)
    #                 embeddings.append(emb.unsqueeze(0))
    #                 # print(vocab[label])

    #             if len(embeddings) >= MAX_SEG_LABEL:
    #                 break

    #         embeddings.extend([torch.zeros_like(label_emb[0],device=label_emb.device).unsqueeze(0) for _ in range(MAX_SEG_LABEL - len(embeddings))])
    #         embeddings = torch.cat(embeddings, dim=0)
    #         batch_embeddings.append(embeddings.unsqueeze(0))

    #     batch_embeddings = torch.cat(batch_embeddings, dim=0)

    #     return batch_embeddings

    # def visualize(
    #     self,
    #     image: Image.Image,
    #     sem_seg: np.ndarray,
    #     vocabulary: List[str],
    #     output_file: str = None,
    #     mode: str = "overlay",
    # ) -> Union[Image.Image, None]:
    #     """
    #     Visualize the segmentation result.
    #     Args:
    #         image (Image.Image): the input image
    #         sem_seg (np.ndarray): the segmentation result
    #         vocabulary (List[str]): the vocabulary used for the segmentation
    #         output_file (str): the output file path
    #         mode (str): the visualization mode, can be "overlay" or "mask"
    #     Returns:
    #         Image.Image: the visualization result. If output_file is not None, return None.
    #     """
    #     # add temporary metadata
    #     # set numpy seed to make sure the colors are the same
    #     np.random.seed(0)
    #     colors = [random_color(rgb=True, maximum=255) for _ in range(len(vocabulary))]
    #     MetadataCatalog.get("_temp").set(stuff_classes=vocabulary, stuff_colors=colors)
    #     metadata = MetadataCatalog.get("_temp")
    #     if isinstance(image, np.ndarray):
    #         image = Image.fromarray(image)

    #     image.save(output_file + "_original.png")
    #     if mode == "overlay":
    #         v = Visualizer(image, metadata)
    #         v = v.draw_sem_seg(sem_seg, area_threshold=0).get_image()
    #         v = Image.fromarray(v)
    #     else:
    #         v = np.zeros((image.size[1], image.size[0], 3), dtype=np.uint8)
    #         labels, areas = np.unique(sem_seg, return_counts=True)
    #         sorted_idxs = np.argsort(-areas).tolist()
    #         labels = labels[sorted_idxs]
    #         for label in filter(lambda l: l < len(metadata.stuff_classes), labels):
    #             v[sem_seg == label] = metadata.stuff_colors[label]
    #         v = Image.fromarray(v)
    #     # remove temporary metadata
    #     MetadataCatalog.remove("_temp")
    #     if output_file is None:
    #         return v
    #     v.save(output_file)

    def masked_global_average_pooling(self, input_tensor, mask, idf=None):
        mask = mask.logical_not() # mask[x] = input[x] is not pad
        mask_expanded = mask.unsqueeze(-1).expand_as(input_tensor).float()
        input_tensor_masked = input_tensor * mask_expanded
        num_elements = mask.sum(dim=1,keepdim=True).float() # TODO: チェック
        if idf is not None:
            idf = idf.unsqueeze(-1).expand_as(input_tensor_masked)
            input_tensor_masked = input_tensor_masked * idf

        output_tensor = input_tensor_masked.sum(dim=1) / num_elements
        return output_tensor


    def forward(
        self,
        src_tokens: torch.tensor,
        mt_tokens: torch.tensor,
        ref_tokens: torch.tensor,
        src_lengths: torch.tensor,
        mt_lengths: torch.tensor,
        ref_lengths: torch.tensor,
        src_idf: torch.tensor,
        mt_idf: torch.tensor,
        ref_idf: torch.tensor,
        src: torch.tensor,
        imgs: torch.tensor,
        alt_tokens: torch.tensor = None,
        alt_lengths: torch.tensor = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Function that encodes both Source, MT and Reference and returns a quality score.

        :param src_tokens: SRC sequences [batch_size x src_seq_len]
        :param mt_tokens: MT sequences [batch_size x mt_seq_len]
        :param ref_tokens: REF sequences [batch_size x ref_seq_len]
        :param src_lengths: SRC lengths [batch_size]
        :param mt_lengths: MT lengths [batch_size]
        :param ref_lengths: REF lengths [batch_size]

        :param alt_tokens: Alternative REF sequences [batch_size x alt_seq_len]
        :param alt_lengths: Alternative REF lengths [batch_size]

        :return: Dictionary with model outputs to be passed to the loss function.
        """

        # images = [cv2.resize(img, dsize=(512, 512)) for img in imgs]
        # self.san.eval()
        # vocabulary = self.vocabulary
        # with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        #     # Apply pre-processing to image.
        #     inputs = []
        #     for image in images:  # TODO: RGB ? BGR ?
        #         height, width = image.shape[:2]
        #         image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        #         inputs.append({"image": image, "height": height, "width": width, "vocabulary": vocabulary})

        #     # print("vocabulary:", vocabulary)
        #     results = self.san(inputs)

        # seg_map = [res["sem_seg"].argmax(dim=0).unsqueeze(0) for res in results]
        # pred = torch.cat(seg_map,dim=0) # (B H W)

        # if VISUALIZE:
        #     for b in range(pred.shape[0]):
        #         self.visualize(images[b], pred[b].cpu().numpy(), vocabulary, output_file=f"logs/output_{b}.png")

        # labels = self.encoder.prepare_sample(vocabulary)
        # label_emb = self.get_sentence_embedding(labels["tokens"].cuda(), labels["lengths"].cuda())

        # seg_emb = self.create_embeddings_from_mask(pred, len(vocabulary), label_emb)

        # pred_patch = self.patchify(pred.float())
        # pred_patch = self.patch_linear(pred_patch)

        _, src_sentembs, src_mask, padding_index = self.get_sentence_embedding(src_tokens, src_lengths,pooling=False)
        _, mt_sentembs, mt_mask, _ = self.get_sentence_embedding(mt_tokens, mt_lengths,pooling=False)
        _, ref_sentembs, ref_mask, _ = self.get_sentence_embedding(ref_tokens, ref_lengths,pooling=False)

        # original comet
        src_sentemb = self.masked_global_average_pooling(src_sentembs,src_mask.logical_not(),src_idf)
        mt_sentemb = self.masked_global_average_pooling(mt_sentembs, mt_mask.logical_not(),mt_idf)
        ref_sentemb = self.masked_global_average_pooling(ref_sentembs, ref_mask.logical_not(), ref_idf)

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb *  src_sentemb

        embedded_sequences = torch.cat(
            (src_sentemb, mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
        )

        # word unit
        # src_idx, mt_idx, ref_idx = np.cumsum([s.shape[1] for s in [src_sentembs,mt_sentembs,ref_sentembs]])
        # x = torch.cat([src_sentembs,mt_sentembs,ref_sentembs], dim=1)
        # padding_mask = torch.cat([src_mask, mt_mask, ref_mask], dim=1)
        # padding_mask = padding_mask.logical_not() # invert mask
        # x = self.transformer(x, src_key_padding_mask=padding_mask)

        # src_sentemb = self.masked_global_average_pooling(x[:,:src_idx,:], padding_mask[:,:src_idx],src_idf)
        # mt_sentemb = self.masked_global_average_pooling(x[:,src_idx:mt_idx,:], padding_mask[:,src_idx:mt_idx],mt_idf)
        # ref_sentemb = self.masked_global_average_pooling(x[:,mt_idx:ref_idx,:], padding_mask[:,mt_idx:ref_idx],ref_idf)

        # diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        # diff_src = torch.abs(mt_sentemb - src_sentemb)

        # prod_ref = mt_sentemb * ref_sentemb
        # prod_src = mt_sentemb * src_sentemb

        if (
            not hasattr(
                self.hparams, "switch_prob"
            )  # compatability with older checkpoints!
            or self.hparams.switch_prob <= 0.0
        ):
            # embedded_sequences = torch.cat(
            #     (embedded_sequences, mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
            # )
            score = self.ff(embedded_sequences)

            if (alt_tokens is not None) and (alt_lengths is not None):
                assert False
                alt_sentemb = self.get_sentence_embedding(alt_tokens, alt_lengths)

                diff_alt = torch.abs(mt_sentemb - alt_sentemb)
                prod_alt = mt_sentemb * alt_sentemb

                embedded_sequences = torch.cat(
                    (mt_sentemb, alt_sentemb, prod_alt, diff_alt, prod_src, diff_src),
                    dim=1,
                )
                score = (score + self.ff(embedded_sequences)) / 2

            return {"score": score}

        if self.training:
            switch = random.random() < self.hparams.switch_prob

            if switch:
                embedded_sequences = torch.cat(
                    (mt_sentemb, ref_sentemb, prod_src, diff_src, prod_ref, diff_ref,pred_patch),
                    dim=1,
                )
            else:
                embedded_sequences = torch.cat(
                    (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src,pred_patch),
                    dim=1,
                )
            return {"score": self.ff(embedded_sequences)}

        elif (alt_tokens is not None) and (alt_lengths is not None):
            assert False
            # Switcheroo Inference!
            alt_sentemb = self.get_sentence_embedding(alt_tokens, alt_lengths)
            diff_alt = torch.abs(mt_sentemb - alt_sentemb)
            prod_alt = mt_sentemb * alt_sentemb

            # Source + MT + Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
            )
            src_mt_ref = self.ff(embedded_sequences)

            # Reference + MT + Source
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_src, diff_src, prod_ref, diff_ref), dim=1
            )
            ref_mt_src = self.ff(embedded_sequences)

            # Source + MT + Alternative Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, alt_sentemb, prod_alt, diff_alt, prod_src, diff_src), dim=1
            )
            src_mt_alt = self.ff(embedded_sequences)

            # Alternative Reference + MT + Source
            embedded_sequences = torch.cat(
                (mt_sentemb, alt_sentemb, prod_src, diff_src, prod_alt, diff_alt), dim=1
            )
            alt_mt_src = self.ff(embedded_sequences)

            # Alternative Reference + MT + Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, alt_sentemb, prod_alt, diff_alt, prod_ref, diff_ref), dim=1
            )
            alt_mt_ref = self.ff(embedded_sequences)

            # Reference + MT + Alternative Reference
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_alt, diff_alt), dim=1
            )
            ref_mt_alt = self.ff(embedded_sequences)

            score = torch.stack(
                [src_mt_ref, ref_mt_src, src_mt_alt, alt_mt_src, alt_mt_ref, ref_mt_alt]
            )
            confidence = 1 - score.std(dim=0)

            return {"score": score.mean(dim=0) * confidence, "confidence": confidence}

        else:
            assert False
            # Usual scoring
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src), dim=1
            )
            score = self.ff(embedded_sequences) * (1 - self.hparams.switch_prob)

            # Switch src and reference embeddings
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_src, diff_src, prod_ref, diff_ref), dim=1
            )
            return {
                "score": score + self.ff(embedded_sequences) * self.hparams.switch_prob
            }

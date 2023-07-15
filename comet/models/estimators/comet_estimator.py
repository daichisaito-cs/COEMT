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
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.engine import DefaultTrainer
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.visualizer import Visualizer, random_color
from huggingface_hub import hf_hub_download
from PIL import Image

from san import add_san_config
from san.data.datasets.register_coco_stuff_164k import COCO_CATEGORIES

BACKBONE = "san_vit_b_16"
model_cfg = {
    "san_vit_b_16": {
        "config_file": "configs/san_clip_vit_res4_coco.yaml",
        "model_path": "huggingface:san_vit_b_16.pth",
    },
    "san_vit_large_16": {
        "config_file": "configs/san_clip_vit_large_res4_coco.yaml",
        "model_path": "huggingface:san_vit_large_14.pth",
    },
}


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


def setup(device=None):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_san_config(cfg)
    cfg.merge_from_file(model_cfg[BACKBONE]["config_file"])
    cfg.MODEL.DEVICE = device or "cuda" if torch.cuda.is_available() else "cpu"
    cfg.freeze()
    return cfg

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
            self.encoder.output_units * (6+64)
            if self.hparams.pool != "cls+avg"
            else self.encoder.output_units * 2 * (6+64)
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

        self.patch_linear = torch.nn.Linear(1024,768)


        model_path = model_cfg[BACKBONE]["model_path"]
        cfg = setup()
        self.san = DefaultTrainer.build_model(cfg)
        if model_path.startswith("huggingface:"):
            model_path = download_model(model_path)
        print("Loading model from: ", model_path)
        DetectionCheckpointer(self.san, save_dir=cfg.OUTPUT_DIR).resume_or_load(model_path)
        print("Loaded model from: ", model_path)


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

        # i = self.new_processor(sample["src"], images=sample["img"], return_tensors="pt", padding=True)
        # i = {k : v.cuda() for k,v in i.items() }
        # o = self.clip_new_model(**i)
        inputs["imgs"] = sample["img"]
        inputs["src"] = sample["src"]
        # inputs["imgs"] = o.image_embeds

        # inputs["imgs"] = sample["img"]
        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

    def patchify(self, img, patch_size=32):
        assert len(img.shape) == 3, "3D tensors expected"
        b, h, w = img.shape
        assert w % patch_size == 0 and h % patch_size == 0
            
        unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
        patches = unfold(img.unsqueeze(1))
        patches = patches.permute(0, 2, 1) # b, c, n
        return patches

    def _merge_vocabulary(self, vocabulary: List[str]) -> List[str]:
        default_voc = [c["name"] for c in COCO_CATEGORIES]
        return vocabulary + [c for c in default_voc if c not in vocabulary]

    def _postprocess(
        self, result: torch.Tensor, ori_vocabulary: List[str]
    ):
        result = result.argmax(dim=0)  # (H, W)
        if len(ori_vocabulary) == 0:
            return result
        result[result >= len(ori_vocabulary)] = len(ori_vocabulary)
        return result

    def forward(
        self,
        src_tokens: torch.tensor,
        mt_tokens: torch.tensor,
        ref_tokens: torch.tensor,
        src_lengths: torch.tensor,
        mt_lengths: torch.tensor,
        ref_lengths: torch.tensor,
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

        # TODO: 修正img = read_image(path, format="BGR")
        vocabulary = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
        vocabulary = list(set([v.lower().strip() for v in vocabulary]))

        images = [cv2.resize(img, dsize=(256, 256)) for img in imgs]
        self.san.eval()
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for image in images:  # TODO: RGB ? BGR ?
                height, width = image.shape[:2]
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width, "vocabulary": vocabulary})
            
            # print("vocabulary:", vocabulary)
            ori_vocabulary = vocabulary
            vocabulary = self._merge_vocabulary(vocabulary)
            results = self.san(inputs)
        
        seg_map = [self._postprocess(res["sem_seg"], ori_vocabulary).unsqueeze(0) for res in results]
        pred = torch.cat(seg_map,dim=0).float()
        pred_patch = self.patchify(pred)
        pred_patch = self.patch_linear(pred_patch)
        # pred_patch = torch.randn((src_lengths.shape[0],64,768)).float().cuda() # for DEBUG
        
        need_vis = False
        if need_vis:
            import copy
            blank_area = (r[0] == 0)
            mask = copy.deepcopy(pred_mask.to('cpu'))
            mask[blank_area] = 255
            mask = np.array(mask, dtype=np.int)

            visualizer = OVSegVisualizer(image, self.metadata, instance_mode=self.instance_mode, class_names=class_names)
            vis_output = visualizer.draw_sem_seg(mask)
        
        src_sentemb = self.get_sentence_embedding(src_tokens, src_lengths)
        mt_sentemb = self.get_sentence_embedding(mt_tokens, mt_lengths)
        ref_sentemb = self.get_sentence_embedding(ref_tokens, ref_lengths)

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb * src_sentemb

        mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src \
                                    = [x.unsqueeze(1) for x in [mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src]]

        if (
            not hasattr(
                self.hparams, "switch_prob"
            )  # compatability with older checkpoints!
            or self.hparams.switch_prob <= 0.0
        ):
            embedded_sequences = torch.cat(
                (mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src,pred_patch), dim=1
            )
            embedded_sequences = embedded_sequences.flatten(1) # TODO: 修正
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

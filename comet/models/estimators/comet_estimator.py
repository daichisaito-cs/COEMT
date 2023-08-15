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
import clip

from typing import List, Union
# from skimage import measure

import numpy as np
import torch.nn.functional as F

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
            self.encoder.output_units * 18
            if self.hparams.pool != "cls+avg"
            else self.encoder.output_units * 2 * 18
        )

        self.ff = torch.nn.Sequential(*[
            FeedForward(
                in_dim=input_emb_sz,
                out_dim=1,
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

        self.clip_linear = torch.nn.Linear(512, 768)
        self.clip_model, self.preprocess = clip.load("ViT-B/16")


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

        # one_sample = self.encoder.tokenizer.batch_decode(src_inputs["src_tokens"],src_inputs["src_lengths"])[0]
        # one_sample_idf = inputs["src_idf"][0]
        # print(one_sample)
        # print(one_sample_idf)

        if inference:
            return inputs

        targets = {"score": torch.tensor(sample["score"], dtype=torch.float)}
        return inputs, targets

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

        imgs = [self.preprocess(img).unsqueeze(0).cuda() for img in imgs]
        imgs = torch.cat(imgs, dim=0)

        with torch.no_grad():
            img_emb = self.clip_model.encode_image(imgs).float()

        img_emb = self.clip_linear(img_emb)

        src_sentemb_org, src_sentembs, src_mask, padding_index = self.get_sentence_embedding(src_tokens, src_lengths,pooling=False)
        mt_sentemb_org, mt_sentembs, mt_mask, _ = self.get_sentence_embedding(mt_tokens, mt_lengths,pooling=False)
        ref_sentemb_org, ref_sentembs, ref_mask, _ = self.get_sentence_embedding(ref_tokens, ref_lengths,pooling=False)

        src_sentemb = self.masked_global_average_pooling(src_sentembs,src_mask.logical_not(), src_idf)
        mt_sentemb = self.masked_global_average_pooling(mt_sentembs, mt_mask.logical_not(), mt_idf)
        ref_sentemb = self.masked_global_average_pooling(ref_sentembs, ref_mask.logical_not(), ref_idf)

        diff_ref = torch.abs(mt_sentemb - ref_sentemb)
        diff_src = torch.abs(mt_sentemb - src_sentemb)
        diff_img = torch.abs(mt_sentemb - img_emb)

        prod_ref = mt_sentemb * ref_sentemb
        prod_src = mt_sentemb *  src_sentemb
        prod_img = mt_sentemb *  img_emb

        # original comet

        diff_ref_org = torch.abs(mt_sentemb_org - ref_sentemb_org)
        diff_src_org = torch.abs(mt_sentemb_org - src_sentemb_org)
        diff_img_org = torch.abs(mt_sentemb_org - img_emb)

        prod_ref_org = mt_sentemb_org * ref_sentemb_org
        prod_src_org = mt_sentemb_org *  src_sentemb_org
        prod_img_org = mt_sentemb_org *  img_emb

        embs = torch.cat(
            (src_sentemb_org, mt_sentemb_org, ref_sentemb_org, prod_ref_org, diff_ref_org, prod_src_org, diff_src_org, prod_img_org, diff_img_org), dim=1
        )

        if (
            not hasattr(
                self.hparams, "switch_prob"
            )  # compatability with older checkpoints!
            or self.hparams.switch_prob <= 0.0
        ):
            embedded_sequences = torch.cat(
                (src_sentemb, mt_sentemb, ref_sentemb, prod_ref, diff_ref, prod_src, diff_src, prod_img, diff_img, embs), dim=1
            )
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

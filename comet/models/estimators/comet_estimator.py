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

from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer


from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from ovseg.open_vocab_seg import add_ovseg_config
from detectron2.config import get_cfg

from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer



class OVSegPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super().__init__(cfg)
        # freeze model
        for param in self.model.parameters():
            param.requires_grad = False

    def __call__(self, images, class_names):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            # Apply pre-processing to image.
            inputs = []
            for original_image in images:
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                inputs.append({"image": image, "height": height, "width": width, "class_names": class_names})
            
            predictions = self.model(inputs)
            return predictions


def setup_cfg():
    cfg = get_cfg()
    # for poly lr schedule
    add_deeplab_config(cfg)
    add_ovseg_config(cfg)
    cfg.merge_from_file("ovseg/configs/ovseg_swinB_vitL_demo.yaml")
    cfg.merge_from_list("MODEL.WEIGHTS ovseg/ovseg_swinbase_vitL14_ft_mpt.pth".split())
    cfg.freeze()
    return cfg





class OVSegVisualizer(Visualizer):
    def __init__(self, img_rgb, metadata=None, scale=1.0, instance_mode=ColorMode.IMAGE, class_names=None):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self.class_names = class_names

    def draw_sem_seg(self, sem_seg, area_threshold=None, alpha=0.8):
        """
        Draw semantic segmentation predictions/labels.

        Args:
            sem_seg (Tensor or ndarray): the segmentation of shape (H, W).
                Each value is the integer label of the pixel.
            area_threshold (int): segments with less than `area_threshold` are not drawn.
            alpha (float): the larger it is, the more opaque the segmentations are.

        Returns:
            output (VisImage): image object with visualizations.
        """
        if isinstance(sem_seg, torch.Tensor):
            sem_seg = sem_seg.numpy()
        labels, areas = np.unique(sem_seg, return_counts=True)
        sorted_idxs = np.argsort(-areas).tolist()
        labels = labels[sorted_idxs]
        class_names = self.class_names if self.class_names is not None else self.metadata.stuff_classes

        for label in filter(lambda l: l < len(class_names), labels):
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[label]]
            except (AttributeError, IndexError):
                mask_color = None

            binary_mask = (sem_seg == label).astype(np.uint8)
            text = class_names[label]
            self.draw_binary_mask(
                binary_mask,
                color=mask_color,
                edge_color=(1.0, 1.0, 240.0 / 255),
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )
        return self.output


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

        cfg = setup_cfg()
        self.predictor = OVSegPredictor(cfg)
        self.patch_linear = torch.nn.Linear(1024,768)

        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )

        self.cpu_device = torch.device("cpu")
        self.instance_mode = ColorMode.IMAGE


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
        class_names = ['person', 'bicycle', 'car', 'motorbike', 'aeroplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'sofa', 'pottedplant', 'bed', 'diningtable', 'toilet', 'tvmonitor', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

        imgs = [cv2.resize(img, dsize=(256, 256)) for img in imgs]
        predictions = self.predictor(imgs, class_names)
        pred = [p["sem_seg"].argmax(dim=0).unsqueeze(0) for p in predictions]
        pred = torch.cat(pred,dim=0).float() # B, H, W
        pred_patch = self.patchify(pred)
        pred_patch = self.patch_linear(pred_patch)
        
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

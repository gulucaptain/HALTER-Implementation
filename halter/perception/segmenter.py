"""Open-vocabulary segmentation of task-relevant objects (GroundedSAM).

GroundedSAMSegmenter = Grounding DINO (text-prompted boxes) + SAM / SAM2 (box-prompted masks),
both loaded through HuggingFace `transformers` so no extra repo checkout is needed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Detection:
    label: str
    mask: np.ndarray       # (H, W) bool
    score: float
    box: Optional[np.ndarray] = None  # xyxy


class Segmenter(ABC):
    @abstractmethod
    def segment(self, rgb: np.ndarray, labels: List[str]) -> List[Detection]: ...


class GroundedSAMSegmenter(Segmenter):
    def __init__(
        self,
        dino_id: str = "IDEA-Research/grounding-dino-base",
        sam_id: str = "facebook/sam-vit-huge",
        device: str = "cuda",
        box_threshold: float = 0.30,
        text_threshold: float = 0.25,
        aliases: Optional[Dict[str, str]] = None,
    ):
        import torch
        from transformers import (AutoModelForZeroShotObjectDetection, AutoProcessor,
                                  SamModel, SamProcessor)
        self.torch = torch
        self.device = device
        self.dino_proc = AutoProcessor.from_pretrained(dino_id)
        self.dino = AutoModelForZeroShotObjectDetection.from_pretrained(dino_id).to(device).eval()
        self.sam_proc = SamProcessor.from_pretrained(sam_id)
        self.sam = SamModel.from_pretrained(sam_id).to(device).eval()
        self.box_threshold, self.text_threshold = box_threshold, text_threshold
        # object name in the graph -> phrase given to Grounding DINO ("lower_drawer" -> "drawer handle")
        self.aliases = aliases or {}

    def _phrase(self, label: str) -> str:
        return self.aliases.get(label, label.replace("_", " "))

    def segment(self, rgb: np.ndarray, labels: List[str]) -> List[Detection]:
        from PIL import Image
        torch = self.torch
        img = Image.fromarray(rgb)
        phrases = [self._phrase(l) for l in labels]
        text = ". ".join(phrases) + "."
        inp = self.dino_proc(images=img, text=text, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.dino(**inp)
        res = self.dino_proc.post_process_grounded_object_detection(
            out, inp.input_ids, threshold=self.box_threshold, text_threshold=self.text_threshold,
            target_sizes=[img.size[::-1]])[0]
        # keep the best box per requested label
        best: Dict[str, tuple] = {}
        text_labels = res.get("text_labels", res.get("labels"))
        for box, score, lab in zip(res["boxes"], res["scores"], text_labels):
            lab = str(lab)
            for name, ph in zip(labels, phrases):
                if ph in lab or lab in ph:
                    if name not in best or float(score) > best[name][1]:
                        best[name] = (box.cpu().numpy(), float(score))
        if not best:
            return []
        names = list(best)
        boxes = [[best[n][0].tolist() for n in names]]
        sin = self.sam_proc(img, input_boxes=boxes, return_tensors="pt").to(self.device)
        with torch.no_grad():
            sout = self.sam(**sin, multimask_output=False)
        masks = self.sam_proc.image_processor.post_process_masks(
            sout.pred_masks.cpu(), sin["original_sizes"].cpu(), sin["reshaped_input_sizes"].cpu())[0]
        dets = []
        for i, n in enumerate(names):
            dets.append(Detection(n, masks[i, 0].numpy().astype(bool), best[n][1], best[n][0]))
        return dets


class ColorThresholdSegmenter(Segmenter):
    """Cheap fallback for debugging: each label maps to an RGB range {label: [[rmin,gmin,bmin],[rmax,gmax,bmax]]}."""

    def __init__(self, ranges: Dict[str, List[List[int]]]):
        self.ranges = {k: (np.array(v[0]), np.array(v[1])) for k, v in ranges.items()}

    def segment(self, rgb: np.ndarray, labels: List[str]) -> List[Detection]:
        dets = []
        for l in labels:
            if l not in self.ranges:
                continue
            lo, hi = self.ranges[l]
            m = np.all((rgb >= lo) & (rgb <= hi), axis=-1)
            if m.sum() > 50:
                dets.append(Detection(l, m, 1.0))
        return dets

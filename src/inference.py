# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
from dataclasses import dataclass
import json as json_mod

# --- Hierarchical Logic & Calibration ---

def apply_temperature_scaling(logits, labels):
    """Finds the optimal temperature for calibrating model confidence."""
    logits_tensor = torch.tensor(logits, dtype=torch.float32)
    labels_tensor = torch.tensor(labels, dtype=torch.long)

    class TemperatureScaler(nn.Module):
        def __init__(self):
            super().__init__()
            self.temperature = nn.Parameter(torch.ones(1) * 1.5)

        def forward(self, logits):
            return logits / self.temperature

    model = TemperatureScaler()
    optimizer = torch.optim.LBFGS([model.temperature], lr=0.01, max_iter=50)

    def eval_fn():
        optimizer.zero_grad()
        loss = F.cross_entropy(model(logits_tensor), labels_tensor)
        loss.backward()
        return loss

    optimizer.step(eval_fn)
    return model.temperature.item()

def hierarchical_predict(image_paths, model_s1, model_s2, processor, device, 
                         batch_size=32, calib_path=None, label2id_s1=None, id2label_s2=None):
    """
    Two-step prediction pipeline:
    1. Relevance Filter (Stage 1)
    2. Emotion Classifier (Stage 2)
    """
    results = []
    
    # Load Calibration for S1 
    T_s1, tau = 1.0, 0.30
    if calib_path and os.path.exists(calib_path):
        with open(calib_path, "r") as f:
            _c = json_mod.load(f)
            T_s1 = float(_c.get("T", 1.0))
            tau  = float(_c.get("tau", 0.30))

    @dataclass
    class _Thresh:
        base_conf: float = 0.65
        entropy_max: float = 1.60
        minority_classes: tuple = ("sadness", "speech_action")
        minority_conf: float = 0.90
    
    thr_cfg = _Thresh()

    for i in tqdm(range(0, len(image_paths), batch_size), desc="🔬 Hierarchical Inference"):
        batch_paths = image_paths[i:i+batch_size]
        images, valid_paths = [], []
        
        for path in batch_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                valid_paths.append(path)
            except Exception:
                continue

        if not images: continue

        inputs = processor(images=images, return_tensors="pt").to(device)

        with torch.no_grad():
            # Stage 1: Calibration + Gate 
            logits_s1 = model_s1(**inputs).logits / max(T_s1, 1e-3)
            probs_s1 = F.softmax(logits_s1, dim=-1)
            relevant_mask = (probs_s1[:, label2id_s1['relevant']] >= tau)
            
            # Stage 2: Emotion 
            if relevant_mask.any():
                relevant_inputs = {k: v[relevant_mask] for k, v in inputs.items()}
                logits_s2 = model_s2(**relevant_inputs).logits
                probs_s2 = F.softmax(logits_s2, dim=-1)
                confs_s2, preds_s2 = torch.max(probs_s2, dim=-1)
                entropies_s2 = (-probs_s2 * torch.log(probs_s2 + 1e-12)).sum(dim=1)

        # Mapping Results 
        s2_idx = 0
        for j in range(len(valid_paths)):
            if relevant_mask[j]:
                label_idx = preds_s2[s2_idx].item()
                label_name = id2label_s2[label_idx]
                conf, ent = float(confs_s2[s2_idx].item()), float(entropies_s2[s2_idx].item())
                s2_idx += 1

                # Review routing 
                thr = thr_cfg.minority_conf if label_name in thr_cfg.minority_classes else thr_cfg.base_conf
                final_label = "review_lowconf" if (conf < thr or ent > thr_cfg.entropy_max) else label_name
                results.append({"image_path": valid_paths[j], "prediction": final_label, "confidence": conf})
            else:
                results.append({"image_path": valid_paths[j], "prediction": "irrelevant", "confidence": float(probs_s1[j, 1].item())})
                
    return results
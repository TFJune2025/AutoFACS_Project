# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer

# --- Stage 1: Relevance Focal Loss ---
# Handles binary relevance filtering by emphasizing difficult positives.
class RelevantFocalCrossEntropy(nn.Module):
    def __init__(self, class_weights: torch.Tensor, gamma: float = 2.0, relevant_id: int = 1):
        """
        Computes focal-modulated cross-entropy for the relevance filter.
        """
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights, reduction="none")
        self.gamma = gamma
        self.relevant_id = relevant_id

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        with torch.no_grad():
            probs = torch.softmax(logits, dim=-1)
            p_t = probs[torch.arange(probs.size(0)), targets]

        mask = (targets == self.relevant_id).float()
        focal = (1.0 - p_t).pow(self.gamma) * mask + (1.0 - mask)
        return (focal * ce).mean()

# --- Stage 2: Targeted Label Smoothing ---
# Prevents over-smoothing on critical "weak" labels like sadness or contempt.
class TargetedSmoothedCrossEntropyLoss(nn.Module):
    def __init__(self, smoothing=0.05, target_class_names=None, label2id_map=None, focal_gamma=None):
        """
        Applies targeted smoothing to encourage confident predictions for specific classes.
        """
        super().__init__()
        self.smoothing = smoothing
        self.focal_gamma = focal_gamma
        if target_class_names and label2id_map:
            self.target_class_ids = [label2id_map[name] for name in target_class_names]
        else:
            self.target_class_ids = [] 

    def forward(self, logits, target):
        num_classes = logits.size(1)
        with torch.no_grad():
            smooth_labels = torch.full_like(logits, self.smoothing / (num_classes - 1)) 
            smooth_labels.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing) 
            if self.target_class_ids:
                target_mask = torch.isin(target, torch.tensor(self.target_class_ids, device=target.device)) 
                if target_mask.any():
                    sharp_labels = F.one_hot(target[target_mask], num_classes=num_classes).float() 
                    smooth_labels[target_mask] = sharp_labels 

        log_probs = F.log_softmax(logits, dim=1)
        ce_per_sample = -(smooth_labels * log_probs).sum(dim=1) 

        if self.focal_gamma is not None and self.focal_gamma > 0:
            with torch.no_grad():
                probs = torch.softmax(logits, dim=1)
                pt = (probs * smooth_labels).sum(dim=1).clamp_min(1e-6) 
            ce_per_sample = ((1 - pt) ** self.focal_gamma) * ce_per_sample 

        return ce_per_sample.mean() 

# --- Custom Orchestrator: Unified Trainer ---
class CustomLossTrainer(Trainer):
    def __init__(self, *args, loss_fct=None, class_weights=None, **kwargs):
        """
        Extension of Hugging Face Trainer to support multi-stage custom losses.
        """
        super().__init__(*args, **kwargs)
        self.loss_fct = loss_fct
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.loss_fct:
            loss = self.loss_fct(logits, labels) 
        else:
            # Fallback for Stage 1 standard weights
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights) 
            loss = loss_fct(logits, labels) 
          
        return (loss, outputs) if return_outputs else loss
# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential.
# ==============================================================================

import torch
import torch.nn.functional as F
from PIL import Image

def predict_emotions(face_image, model, processor, device):
    """Unified logic for predicting emotions from a single PIL image."""
    inputs = processor(images=face_image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        logits = model(**inputs).logits
        probabilities = F.softmax(logits, dim=1).squeeze()

    top_confidence, top_pred_idx = torch.max(probabilities, dim=0)
    top_pred_label = model.config.id2label[top_pred_idx.item()]
    entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-9)).item()
    
    results = {
        "label": top_pred_label,
        "confidence": top_confidence.item(),
        "entropy": entropy
    }
    # Include individual probabilities for fine-grained analysis.
    for i, prob in enumerate(probabilities):
        results[f"prob_{model.config.id2label[i]}"] = prob.item()
    
    return results

def is_high_conviction(results, conf_thresh=0.85, entropy_thresh=0.45, prob_thresh=0.95):
    """VML 'Gold Standard' filter: filters for high-confidence, low-entropy samples."""
    # Check basic confidence and entropy thresholds
    if results['confidence'] <= conf_thresh or results['entropy'] >= entropy_thresh:
        return False
    
    # Ensure at least one emotion has a dominant probability
    prob_values = [v for k, v in results.items() if k.startswith("prob_")]
    if not any(p > prob_thresh for p in prob_values):
        return False
        
    return True
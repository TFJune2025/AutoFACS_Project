# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix

def compute_metrics_with_confusion(eval_pred, label_names, save_dir, stage_name="Stage2", s2_temperature=1.0):
    """
    Computes accuracy, generates a classification report, and saves a 
    confusion matrix heatmap for the specified stage.
    """
    logits, labels = eval_pred
    if stage_name.lower().startswith("stage2") and s2_temperature != 1.0:
        logits = logits / max(1e-6, float(s2_temperature))

    probs = torch.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = probs.argmax(axis=-1)

    # Console Report
    print(f"\n📈 Classification Report for {stage_name}:")
    print(classification_report(labels, preds, target_names=label_names, zero_division=0))

    # Save Confusion Matrix Plot
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {stage_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"confusion_matrix_{stage_name}.png"))
    plt.close()

    return {"accuracy": float((preds == labels).mean())}

def check_deployment_readiness(metrics_df, f1_threshold=0.80):
    """
    Analyzes final epoch metrics to determine if the model meets 
    production standards for all classes.
    """
    print("\n" + "="*60)
    print("  DEPLOYMENT READINESS CHECK")
    print("="*60)
    
    last_epoch = metrics_df.iloc[-1]
    label_names = [col.replace("f1_", "") for col in metrics_df.columns if col.startswith("f1_")]
    
    issues_found = False
    for label in label_names:
        f1_score = last_epoch.get(f"f1_{label}", 0)
        status = "✅" if f1_score >= f1_threshold else "❌"
        if f1_score < f1_threshold: issues_found = True
        print(f"  - {status} {label:<15} | F1-Score: {f1_score:.2f}")
            
    if issues_found:
        print("\n⚠️ Model is NOT ready for production. Some classes fall below threshold.")
    else:
        print("\n🚀 Model meets all minimum F1-score thresholds.")
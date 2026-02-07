# ==============================================================================
# Copyright (c) 2024 Natalya Grokh. All Rights Reserved.
# Proprietary and Confidential. Unauthorized copying of this file, via any 
# medium, is strictly prohibited.
# ==============================================================================
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms import RandAugment

# --- Image Integrity Helpers ---

def is_valid_image(filename):
    """Checks for standard image extensions and skips hidden system files."""
    VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".tif", ".tiff") 
    return filename.lower().endswith(VALID_EXTENSIONS) and not filename.startswith("._") 

def ensure_rgb(img):
    """Normalizes any image-like object to a 3-channel RGB PIL Image."""
    if isinstance(img, Image.Image):
        return img.convert("RGB") 
    arr = np.array(img) 
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1) 
    return Image.fromarray(arr.astype(np.uint8)) 

# --- Specialized Augmentations ---

# Pose/Tilt tolerance to reduce false 'irrelevant' tags on near-frontal faces 
STAGES_1_AUG = T.Compose([
    T.RandomResizedCrop(224, scale=(0.8, 1.0)), 
    T.RandomHorizontalFlip(), 
    T.ColorJitter(0.2, 0.2, 0.2, 0.05), 
    T.RandomPerspective(distortion_scale=0.05, p=0.2), 
    T.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.97, 1.03)), 
])

# Heavy augmentation for minority classes like disgust or contempt 
MINORITY_AUG = T.Compose([
    RandAugment(num_ops=2, magnitude=11), 
    T.RandomResizedCrop(224, scale=(0.7, 1.0)), 
    T.ColorJitter(0.3, 0.3, 0.3, 0.1), 
])

# --- Data Collator Lab ---

class DataCollatorWithAugmentation:
    """Applies class-specific PIL transforms and tensor-level RandomErasing."""
    def __init__(self, processor, augment_dict=None, base_augment=None, 
                 random_erasing_prob=0.10, random_erasing_scale=(0.02, 0.08),
                 skip_erasing_label_ids=None):
        self.processor = processor 
        self.augment_dict = augment_dict or {} 
        self.base_augment = base_augment or T.Compose([T.Resize((224, 224))]) 
        self.random_erasing = T.RandomErasing(p=random_erasing_prob, scale=random_erasing_scale, value="random") if random_erasing_prob > 0 else None 
        self.skip_erasing_label_ids = set(skip_erasing_label_ids or []) 
        self.to_tensor = T.ToTensor() 
        self.to_pil = T.ToPILImage() 

    def __call__(self, features):
        processed_images = []
        for x in features:
            label = x["label"] 
            img = ensure_rgb(x["image"]) 
            pil_aug = self.augment_dict.get(label, self.base_augment) 
            img = pil_aug(img) 
            
            img_t = self.to_tensor(img) 
            if self.random_erasing and label not in self.skip_erasing_label_ids:
                img_t = self.random_erasing(img_t) 
            processed_images.append(self.to_pil(img_t)) 

        batch = self.processor(images=processed_images, return_tensors="pt") 
        batch["labels"] = torch.tensor([x["label"] for x in features], dtype=torch.long) 
        return batch
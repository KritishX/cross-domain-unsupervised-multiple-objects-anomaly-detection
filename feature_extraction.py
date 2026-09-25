
import os
import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
import timm
from tqdm import tqdm
import argparse

# --- Configuration ---
DATASET_DIR = 'dataset'
EMBEDDINGS_DIR = 'embeddings'
IMG_SIZE = 224
DINO_MODEL_NAME = 'vit_small_patch16_224.dino'
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
UPSAMPLE_SIZE = 28     

def get_transforms():
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])

class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Backbone DINO ViT
        self.model = timm.create_model(DINO_MODEL_NAME, pretrained=True, num_classes=0)
        self.model.eval()
        
    def forward(self, x):
        # SIMPLIFICATION: Use ONLY Layer 12 (Stable semantic space)
        # Concatenating layers was increasing noise/sensitivity
        full_out = self.model.forward_features(x) # [B, 197, 384]
        
        cls_token = full_out[:, 0:1, :] # [B, 1, 384]
        patches = full_out[:, 1:, :]   # [B, 196, 384]
        
        B, N, C = patches.shape
        # Spatial interpolation to 28x28
        patches = patches.transpose(1, 2).reshape(B, C, 14, 14)
        patches_up = F.interpolate(patches, size=(UPSAMPLE_SIZE, UPSAMPLE_SIZE), mode='bilinear', align_corners=False)
        patches_up = patches_up.reshape(B, C, -1).transpose(1, 2) # [B, 784, 384]
        
        return torch.cat([cls_token, patches_up], dim=1)

def extract_and_save_features(device, target_categories=None):
    if not os.path.exists(EMBEDDINGS_DIR):
        os.makedirs(EMBEDDINGS_DIR)

    extractor = FeatureExtractor().to(device)
    transform = get_transforms()
    
    if target_categories:
        categories = target_categories
    else:
        categories = sorted([f for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))])
    
    for cat in categories:
        print(f"Extracting features: {cat}")
        cat_embed_dir = os.path.join(EMBEDDINGS_DIR, cat)
        os.makedirs(os.path.join(cat_embed_dir, 'train'), exist_ok=True)
        os.makedirs(os.path.join(cat_embed_dir, 'test'), exist_ok=True)

        # Train
        train_dir = os.path.join(DATASET_DIR, cat, 'train', 'good')
        if os.path.exists(train_dir):
            files = sorted([f for f in os.listdir(train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for f in tqdm(files, desc=f"  Train"):
                path = os.path.join(train_dir, f)
                img = Image.open(path).convert('RGB')
                for angle in [0, 180]:
                    aug_img = img.rotate(angle) if angle != 0 else img
                    img_tensor = transform(aug_img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = extractor(img_tensor)
                        suffix = f"_r{angle}" if angle != 0 else ""
                        save_path = os.path.join(cat_embed_dir, 'train', f.split('.')[0] + suffix + '.pt')
                        torch.save(features.cpu(), save_path)

        # Test
        test_root = os.path.join(DATASET_DIR, cat, 'test')
        if os.path.exists(test_root):
            for test_sub in sorted(os.listdir(test_root)):
                sub_path = os.path.join(test_root, test_sub)
                if not os.path.isdir(sub_path): continue
                os.makedirs(os.path.join(cat_embed_dir, 'test', test_sub), exist_ok=True)
                files = sorted([f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                for f in tqdm(files, desc=f"  Test/{test_sub}"):
                    path = os.path.join(sub_path, f)
                    img = Image.open(path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = extractor(img_tensor)
                        save_path = os.path.join(cat_embed_dir, 'test', test_sub, f.split('.')[0] + '.pt')
                        torch.save(features.cpu(), save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', nargs='+', help='Target categories')
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extract_and_save_features(device, target_categories=args.target)

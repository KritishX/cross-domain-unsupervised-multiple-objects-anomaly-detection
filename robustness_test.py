import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from PIL import Image, ImageEnhance, ImageFilter
from feature_extraction import FeatureExtractor, get_transforms
from anomaly_scoring import CATEGORY_TYPES
from tqdm import tqdm

# --- Configuration ---
MEMORY_BANK_DIR = 'memory_bank'
DATASET_DIR = 'dataset'
OUTPUTS_DIR = 'outputs'

# Categories to test for robustness
TEST_CATS = ['bottle', 'tile', 'cable']

def run_robustness_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    transform = get_transforms()
    
    results = []

    for cat in TEST_CATS:
        print("")
        print(f"Robustness Test for: {cat}")
        # Load Bank
        cat_mem_dir = os.path.join(MEMORY_BANK_DIR, cat)
        bank_path = os.path.join(cat_mem_dir, 'bank.pt')
        if not os.path.exists(bank_path): continue
        
        bank = torch.load(bank_path).to(device)
        pca = joblib.load(os.path.join(cat_mem_dir, 'pca.pkl'))
        cls_bank = torch.load(os.path.join(cat_mem_dir, 'cls_bank.pt')).to(device)
        
        bank = F.normalize(bank, p=2, dim=1)
        cls_bank = F.normalize(cls_bank, p=2, dim=1)

        shifts = [
            ('Clean', None),
            ('Brightness (+50%)', lambda x: ImageEnhance.Brightness(x).enhance(1.5)),
            ('Brightness (-50%)', lambda x: ImageEnhance.Brightness(x).enhance(0.5)),
            ('Blur (G-Blur 2)', lambda x: x.filter(ImageFilter.GaussianBlur(radius=2))),
        ]

        for shift_name, shift_fn in shifts:
            scores = []
            labels = []
            
            test_root = os.path.join(DATASET_DIR, cat, 'test')
            for test_sub in sorted(os.listdir(test_root)):
                sub_path = os.path.join(test_root, test_sub)
                if not os.path.isdir(sub_path): continue
                
                label = 0 if test_sub == 'good' else 1
                files = sorted([f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                
                for f in files:
                    img = Image.open(os.path.join(sub_path, f)).convert('RGB')
                    if shift_fn: img = shift_fn(img)
                    
                    img_tensor = transform(img).unsqueeze(0).to(device)
                    with torch.no_grad():
                        features = extractor(img_tensor) 
                        
                        # Global
                        cls_feat = features[0, 0:1, :]
                        cls_feat = F.normalize(cls_feat, p=2, dim=1)
                        cls_sim = torch.matmul(cls_feat, cls_bank.t())
                        global_score = np.sqrt(max(0.0, 2.0 - 2.0 * torch.max(cls_sim).item()))

                        # Patch
                        patch_features = features[0, 1:, :].cpu().numpy()
                        reduced = pca.transform(patch_features)
                        reduced = torch.from_numpy(reduced).to(device)
                        reduced = F.normalize(reduced, p=2, dim=1)
                        
                        sim = torch.matmul(reduced, bank.t())
                        max_sim, _ = torch.max(sim, dim=1)
                        patch_distances = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
                        
                        k = max(1, int(len(patch_distances) * 0.10))
                        top_k_dists, _ = torch.topk(patch_distances, k)
                        patch_score = torch.mean(top_k_dists).item()
                        
                        final_score = 0.4 * global_score + 0.6 * patch_score
                        scores.append(final_score)
                        labels.append(label)
            
            auroc = roc_auc_score(labels, scores)
            print(f"  {shift_name}: {auroc:.4f}")
            results.append({'category': cat, 'shift': shift_name, 'auroc': auroc})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUTS_DIR, 'robustness_test.csv'), index=False)
    print("")
    print("Robustness Test Complete.")

if __name__ == '__main__':
    run_robustness_test()
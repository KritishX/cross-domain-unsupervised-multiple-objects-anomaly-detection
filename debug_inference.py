import torch
import torch.nn.functional as F
import numpy as np
import joblib
import os
from PIL import Image
from feature_extraction import FeatureExtractor, get_transforms
from config import CATEGORY_TYPES, HYBRID_WEIGHTS, ADAPTIVE_K

def debug_category(category, image_path):
    print("")
    print(f"--- Diagnostic for {category} | Image: {os.path.basename(image_path)} ---")
    
    device = torch.device("cpu")
    extractor = FeatureExtractor().to(device)
    transform = get_transforms()
    
    # Load Assets
    bank_path = f'memory_bank/{category}/bank.pt'
    pca_path = f'memory_bank/{category}/pca.pkl'
    cls_bank_path = f'memory_bank/{category}/cls_bank.pt'
    
    if not os.path.exists(bank_path):
        print(f"Error: Bank not found for {category}")
        return

    bank = torch.load(bank_path, map_location=device)
    pca = joblib.load(pca_path)
    cls_bank = torch.load(cls_bank_path, map_location=device)
    
    bank = F.normalize(bank, p=2, dim=1)
    cls_bank = F.normalize(cls_bank, p=2, dim=1)

    # Process Image
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        features = extractor(img_tensor)
        
        # 1. Global Analysis
        cls_feat = F.normalize(features[0, 0:1, :], p=2, dim=1)
        cls_sim = torch.matmul(cls_feat, cls_bank.t())
        max_cls_sim = torch.max(cls_sim).item()
        global_dist = np.sqrt(max(0.0, 2.0 - 2.0 * max_cls_sim))
        print(f"Global Similarity: {max_cls_sim:.4f} (Distance: {global_dist:.4f})")

        # 2. Patch Analysis
        patch_feats = features[0, 1:, :].numpy()
        reduced = pca.transform(patch_feats)
        reduced = torch.from_numpy(reduced)
        reduced = F.normalize(reduced, p=2, dim=1)
        
        sim = torch.matmul(reduced, bank.t())
        max_sim, _ = torch.max(sim, dim=1)
        patch_dists = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
        
        print(f"Patch Similarity Stats: Min={max_sim.min():.4f}, Max={max_sim.max():.4f}, Mean={max_sim.mean():.4f}")
        print(f"Patch Distance Stats: Min={patch_dists.min():.4f}, Max={patch_dists.max():.4f}, Mean={patch_dists.mean():.4f}")

        # 3. Aggregation Analysis
        cat_type = CATEGORY_TYPES.get(category, 'object')
        k_perc = ADAPTIVE_K[cat_type]
        
        dist_grid = patch_dists.view(1, 1, 28, 28)
        kernel = torch.ones((1, 1, 3, 3)) / 9.0
        smoothed_grid = F.conv2d(dist_grid, kernel, padding=1)
        
        flat_dists = smoothed_grid.view(-1)
        k = max(1, int(len(flat_dists) * k_perc))
        top_k_vals, _ = torch.topk(flat_dists, k)
        patch_score = torch.mean(top_k_vals).item()
        
        final_score = 0.4 * global_dist + 0.6 * patch_score
        print(f"Aggregated Patch Score (Top {k_perc*100}%): {patch_score:.4f}")
        print(f"Final Weighted Score: {final_score:.4f}")

if __name__ == '__main__':
    # Test on a "good" image
    grid_good = 'dataset/grid/test/good/000.png'
    if os.path.exists(grid_good):
        debug_category('grid', grid_good)
    
    # Test on a "defect" image
    grid_bad = 'dataset/grid/test/bent/000.png'
    if os.path.exists(grid_bad):
        debug_category('grid', grid_bad)
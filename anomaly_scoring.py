import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import pandas as pd
from tqdm import tqdm

# --- Configuration ---
EMBEDDINGS_DIR = 'embeddings'
MEMORY_BANK_DIR = 'memory_bank'
OUTPUTS_DIR = 'outputs'

# Meta-info defining which categories are textures vs. discrete objects
# Textures: Repetitive patterns (Grid, Tile, Carpet)
# Objects: Discrete structures (Bottle, Screw, Zipper)
CATEGORY_TYPES = {
    'grid': 'texture', 'carpet': 'texture', 'tile': 'texture', 'wood': 'texture',
    'bottle': 'object', 'capsule': 'object', 'pill': 'object', 'transistor': 'object',
    'metal_nut': 'object', 'screw': 'object', 'zipper': 'object', 'hazelnut': 'object',
    'cable': 'object', 'leather': 'object', 'toothbrush': 'object'
}

def compute_anomaly_scores(target_categories=None):
    """Calculates adaptive image-level anomaly scores using nearest-neighbor search."""
    if not os.path.exists(OUTPUTS_DIR):
        os.makedirs(OUTPUTS_DIR)

    # Categories identified by existing memory banks
    categories = [f for f in os.listdir(MEMORY_BANK_DIR) if os.path.isdir(os.path.join(MEMORY_BANK_DIR, f))]
    results = []

    for cat in categories:
        if target_categories and cat not in target_categories:
            continue
            
        print(f"\nScoring anomalies for category: {cat}")
        cat_mem_dir = os.path.join(MEMORY_BANK_DIR, cat)
        
        # Load the normal coreset bank and PCA model
        bank_path = os.path.join(cat_mem_dir, 'bank.pt')
        pca_path = os.path.join(cat_mem_dir, 'pca.pkl')
        if not os.path.exists(bank_path) or not os.path.exists(pca_path): continue
        
        bank = torch.load(bank_path) 
        pca = joblib.load(pca_path)
        bank = F.normalize(bank, p=2, dim=1) # L2 normalize for stable distance calculation

        # Load the global CLS bank
        cls_bank = None
        cls_bank_path = os.path.join(cat_mem_dir, 'cls_bank.pt')
        if os.path.exists(cls_bank_path):
            cls_bank = torch.load(cls_bank_path)
            cls_bank = F.normalize(cls_bank, p=2, dim=1)

        # Adaptive K: 15% of patches for textures, 6% for objects
        cat_type = CATEGORY_TYPES.get(cat, 'object')
        k_percent = 0.15 if cat_type == 'texture' else 0.06

        test_root = os.path.join(EMBEDDINGS_DIR, cat, 'test')
        if not os.path.exists(test_root): continue
        
        # Loop through testing subdirectories (good, defect_type1, defect_type2...)
        for test_sub in os.listdir(test_root):
            sub_path = os.path.join(test_root, test_sub)
            if not os.path.isdir(sub_path): continue
            
            label = 0 if test_sub == 'good' else 1
            embed_files = [f for f in os.listdir(sub_path) if f.endswith('.pt')]
            
            for f in tqdm(embed_files, desc=f"  Scoring {test_sub}"):
                features = torch.load(os.path.join(sub_path, f)) # Load [1, 1+784, 768]
                
                # --- 1. Global Score (CLS Token) ---
                # Compares global structure against normal images
                global_score = 0.0
                if cls_bank is not None:
                    cls_feat = features[0, 0:1, :]
                    cls_feat = F.normalize(cls_feat, p=2, dim=1)
                    # Matrix multiplication for cosine similarity
                    cls_sim = torch.matmul(cls_feat, cls_bank.t())
                    max_cls_sim = torch.max(cls_sim).item()
                    # Convert cosine similarity to Euclidean distance
                    global_score = np.sqrt(max(0.0, 2.0 - 2.0 * max_cls_sim))

                # --- 2. Patch Score (Local Defects) ---
                # Reduce dims using category PCA
                patch_features = features[0, 1:, :].numpy()
                reduced = pca.transform(patch_features)
                reduced = torch.from_numpy(reduced)
                reduced = F.normalize(reduced, p=2, dim=1)
                
                # Nearest Neighbor search in the bank
                sim = torch.matmul(reduced, bank.t())
                max_sim, _ = torch.max(sim, dim=1) # Highest similarity for each patch
                # Distance: sqrt(2 - 2*cos)
                patch_distances = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
                
                # Spatial Smoothing: Apply 3x3 average blur to the 28x28 distance grid
                dist_grid = patch_distances.view(1, 1, 28, 28)
                kernel = torch.ones((1, 1, 3, 3)) / 9.0
                dist_grid = F.conv2d(dist_grid, kernel, padding=1)
                
                # Image-level aggregation: Mean of the Top-K most anomalous patches
                flat_dists = dist_grid.view(-1)
                k = max(1, int(len(flat_dists) * k_percent))
                top_k_dists, _ = torch.topk(flat_dists, k)
                patch_score = torch.mean(top_k_dists).item()
                
                # Final Image Score: Weighted sum of Global and Local indicators
                final_image_score = 0.4 * global_score + 0.6 * patch_score
                
                results.append({
                    'category': cat, 'image': f.split('.')[0],
                    'label': label, 'defect_type': test_sub,
                    'score': final_image_score
                })

    # Save results to CSV for evaluation
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUTS_DIR, 'anomaly_scores.csv'), index=False)
    print(f"\nScoring complete. Results saved to {OUTPUTS_DIR}/anomaly_scores.csv")

if __name__ == '__main__':
    compute_anomaly_scores()
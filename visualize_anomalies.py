import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import cv2
from PIL import Image
from torchvision import transforms
from feature_extraction import FeatureExtractor, get_transforms
from anomaly_scoring import CATEGORY_TYPES
import matplotlib.pyplot as plt

# --- Configuration ---
MEMORY_BANK_DIR = 'memory_bank'
DATASET_DIR = 'dataset'
OUTPUTS_DIR = 'outputs/heatmaps'
XAI_DIR = 'outputs/explainability'
UPSAMPLE_SIZE = 28

def generate_visualizations(target_categories=None):
    os.makedirs(OUTPUTS_DIR, exist_ok=True)
    os.makedirs(XAI_DIR, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    extractor = FeatureExtractor().to(device)
    transform = get_transforms()
    
    categories = [f for f in os.listdir(MEMORY_BANK_DIR) if os.path.isdir(os.path.join(MEMORY_BANK_DIR, f))]
    
    for cat in categories:
        if target_categories and cat not in target_categories:
            continue
            
        print(f"\nVisualizing anomalies for: {cat}")
        cat_mem_dir = os.path.join(MEMORY_BANK_DIR, cat)
        bank = torch.load(os.path.join(cat_mem_dir, 'bank.pt')) # [5000, 256]
        pca = joblib.load(os.path.join(cat_mem_dir, 'pca.pkl'))
        bank = F.normalize(bank, p=2, dim=1)
        
        cat_type = CATEGORY_TYPES.get(cat, 'object')
        k_percent = 0.15 if cat_type == 'texture' else 0.06

        # Pick 3 anomalous images from test set (one of each defect if possible)
        test_dir = os.path.join(DATASET_DIR, cat, 'test')
        defect_types = [d for d in os.listdir(test_dir) if d != 'good' and os.path.isdir(os.path.join(test_dir, d))]
        
        for dt in defect_types:
            sub_path = os.path.join(test_dir, dt)
            files = [f for f in os.listdir(sub_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not files: continue
            
            # Just take the first image for visualization
            img_name = files[0]
            img_path = os.path.join(sub_path, img_name)
            orig_img = Image.open(img_path).convert('RGB')
            img_tensor = transform(orig_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                features = extractor(img_tensor) # [1, 1+784, 768]
                patch_features = features[0, 1:, :].cpu().numpy()
                reduced = pca.transform(patch_features)
                reduced = torch.from_numpy(reduced)
                reduced = F.normalize(reduced, p=2, dim=1)
                
                # Similarity and Distance
                sim = torch.matmul(reduced, bank.t()) # [784, 5000]
                max_sim, nearest_indices = torch.max(sim, dim=1)
                patch_distances = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
                
                # Reshape and Smooth for Heatmap
                dist_grid = patch_distances.view(1, 1, 28, 28)
                kernel = torch.ones((1, 1, 3, 3)) / 9.0
                dist_grid = F.conv2d(dist_grid, kernel, padding=1)
                heatmap = dist_grid[0, 0].numpy()
                
                # --- Heatmap Generation ---
                # Normalize heatmap for visualization
                v_min, v_max = heatmap.min(), heatmap.max()
                heatmap_norm = (heatmap - v_min) / (v_max - v_min + 1e-8)
                heatmap_norm = (heatmap_norm * 255).astype(np.uint8)
                heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
                heatmap_color = cv2.resize(heatmap_color, (224, 224))
                
                # Overlay
                img_cv = np.array(orig_img.resize((224, 224)))
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)
                overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
                
                save_path = os.path.join(OUTPUTS_DIR, f"{cat}_{dt}_{img_name}")
                cv2.imwrite(save_path, overlay)
                
                # --- Explainability (XAI): Nearest Prototype ---
                # Find the most anomalous patch
                max_idx = torch.argmax(patch_distances).item()
                y, x = divmod(max_idx, 28)
                
                # Get the anomalous patch from the original image (8x8 approximate area in 224x224)
                # 224 / 28 = 8 pixels per patch
                px, py = x * 8, y * 8
                anomaly_patch = img_cv[py:py+16, px:px+16] # Take a slightly larger area for context
                
                # Get top-3 nearest neighbors from the bank
                # Note: Bank features are PCA reduced. We can't easily map back to pixels 
                # unless we stored the patches. But we can describe them.
                # For now, let's visualize the anomaly patch vs the normal image.
                
                fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                axes[0].imshow(cv2.cvtColor(anomaly_patch, cv2.COLOR_BGR2RGB))
                axes[0].set_title(f"Anomalous Patch\n(Score: {patch_distances[max_idx]:.3f})")
                axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
                axes[1].set_title(f"Detected Region in {cat}")
                plt.tight_layout()
                plt.savefig(os.path.join(XAI_DIR, f"{cat}_{dt}_xai.png"))
                plt.close()

if __name__ == '__main__':
    # Visualize hard categories and a few others
    generate_visualizations(target_categories=['grid', 'screw', 'wood', 'capsule', 'bottle'])
import os
import torch
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm
import argparse

# --- Configuration ---
EMBEDDINGS_DIR = 'embeddings'
MEMORY_BANK_DIR = 'memory_bank'
PCA_COMPONENTS = 128
CORESET_SIZE = 5000 

def build_memory_banks(target_categories=None):
    if not os.path.exists(MEMORY_BANK_DIR):
        os.makedirs(MEMORY_BANK_DIR)

    if target_categories:
        categories = target_categories
    else:
        categories = [f for f in os.listdir(EMBEDDINGS_DIR) if os.path.isdir(os.path.join(EMBEDDINGS_DIR, f))]
    
    for cat in categories:
        print(f"\nBuilding memory bank: {cat}")
        train_embed_dir = os.path.join(EMBEDDINGS_DIR, cat, 'train')
        if not os.path.exists(train_embed_dir): continue

        all_features = []
        all_cls = []
        embed_files = [f for f in os.listdir(train_embed_dir) if f.endswith('.pt')]
        
        for f in tqdm(embed_files, desc=f"  Loading"):
            features = torch.load(os.path.join(train_embed_dir, f)) 
            all_cls.append(features[0, 0:1, :].numpy())
            all_features.append(features[0, 1:, :].numpy())

        if not all_features: continue
        pool = np.vstack(all_features)
        cls_pool = np.vstack(all_cls)

        if pool.shape[0] > CORESET_SIZE:
            print(f"  Clustering to {CORESET_SIZE} centroids...")
            kmeans = MiniBatchKMeans(n_clusters=CORESET_SIZE, batch_size=2000, n_init=3, random_state=42)
            kmeans.fit(pool)
            pool = kmeans.cluster_centers_
        
        cat_mem_dir = os.path.join(MEMORY_BANK_DIR, cat)
        os.makedirs(cat_mem_dir, exist_ok=True)
        
        torch.save(torch.from_numpy(cls_pool), os.path.join(cat_mem_dir, 'cls_bank.pt'))
        
        pca = PCA(n_components=min(PCA_COMPONENTS, pool.shape[0], pool.shape[1]))
        reduced_pool = pca.fit_transform(pool)
        
        torch.save(torch.from_numpy(reduced_pool), os.path.join(cat_mem_dir, 'bank.pt'))
        joblib.dump(pca, os.path.join(cat_mem_dir, 'pca.pkl'))
        print(f"  Memory bank saved. Shape: {reduced_pool.shape}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', nargs='+', help='Target categories')
    args = parser.parse_args()
    build_memory_banks(target_categories=args.target)
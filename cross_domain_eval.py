import os
import torch
import torch.nn.functional as F
import numpy as np
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# --- Configuration ---
EMBEDDINGS_DIR = 'embeddings'
MEMORY_BANK_DIR = 'memory_bank'
OUTPUTS_DIR = 'outputs'

# Transfer Pairs: (Source Bank, Target Test Category)
TRANSFER_PAIRS = [
    ('metal_nut', 'screw'),
    ('grid', 'tile'),
    ('wood', 'leather'),
    ('leather', 'wood')
]

def run_transfer_evaluation():
    results = []

    for source_cat, target_cat in TRANSFER_PAIRS:
        print(f"Evaluating Transfer: {source_cat} Bank -> {target_cat} Test")
        
        # Load Source Bank
        source_mem_dir = os.path.join(MEMORY_BANK_DIR, source_cat)
        bank_path = os.path.join(source_mem_dir, 'bank.pt')
        pca_path = os.path.join(source_mem_dir, 'pca.pkl')
        if not os.path.exists(bank_path): continue
        
        bank = torch.load(bank_path)
        pca = joblib.load(pca_path)
        bank = F.normalize(bank, p=2, dim=1)

        # Load Source CLS Bank
        cls_bank_path = os.path.join(source_mem_dir, 'cls_bank.pt')
        cls_bank = torch.load(cls_bank_path)
        cls_bank = F.normalize(cls_bank, p=2, dim=1)

        # Evaluate on Target Test
        target_embed_root = os.path.join(EMBEDDINGS_DIR, target_cat, 'test')
        if not os.path.exists(target_embed_root): continue
        
        pair_scores = []
        pair_labels = []

        for test_sub in os.listdir(target_embed_root):
            sub_path = os.path.join(target_embed_root, test_sub)
            if not os.path.isdir(sub_path): continue
            
            label = 0 if test_sub == 'good' else 1
            files = [f for f in os.listdir(sub_path) if f.endswith('.pt')]
            
            for f in files:
                features = torch.load(os.path.join(sub_path, f)) 
                
                # Global
                cls_feat = features[0, 0:1, :]
                cls_feat = F.normalize(cls_feat, p=2, dim=1)
                cls_sim = torch.matmul(cls_feat, cls_bank.t())
                global_score = np.sqrt(max(0.0, 2.0 - 2.0 * torch.max(cls_sim).item()))

                # Patch
                patch_features = features[0, 1:, :].numpy()
                reduced = pca.transform(patch_features)
                reduced = torch.from_numpy(reduced)
                reduced = F.normalize(reduced, p=2, dim=1)
                
                sim = torch.matmul(reduced, bank.t())
                max_sim, _ = torch.max(sim, dim=1)
                patch_distances = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
                
                # Top-10% mean
                k = max(1, int(len(patch_distances) * 0.10))
                top_k_dists, _ = torch.topk(patch_distances, k)
                patch_score = torch.mean(top_k_dists).item()
                
                final_score = 0.4 * global_score + 0.6 * patch_score
                
                pair_scores.append(final_score)
                pair_labels.append(label)

        if pair_labels:
            auroc = roc_auc_score(pair_labels, pair_scores)
            print(f"  Transfer AUROC: {auroc:.4f}")
            results.append({'pair': f"{source_cat}->{target_cat}", 'auroc': auroc})

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUTPUTS_DIR, 'transfer_evaluation.csv'), index=False)
    print("\nTransfer Evaluation Complete.")

if __name__ == '__main__':
    run_transfer_evaluation()
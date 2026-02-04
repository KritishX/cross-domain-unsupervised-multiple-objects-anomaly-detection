import os
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import timm
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

# --- Configuration ---
DATASET_DIR = 'dataset'
IMG_SIZE = 224
PATCH_SIZE = 16
# ViT-DINO pretrained on ImageNet
DINO_MODEL_NAME = 'vit_small_patch16_224.dino'
# Normalization stats for ImageNet
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
# Optional: Dimensionality reduction for feature bank
APPLY_PCA = True
PCA_COMPONENTS = 128
OUTPUT_DIR = 'outputs'

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- 1. Preprocessing Images for ViT Input ---
def get_transforms(is_train=False):
    transform_list = [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
    ]
    if is_train:
        transform_list.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
        ])
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD)
    ])
    return transforms.Compose(transform_list)

# --- 5. Load Pretrained ViT-DINO Model ---
def load_dino_model(model_name=DINO_MODEL_NAME):
    print(f"Loading model: {model_name}")
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    print("Model loaded successfully.")
    return model

# --- 6. Build Feature Memory Bank ---
def build_feature_memory_bank(model, category_train_dir, transform):
    features = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    image_files = [os.path.join(category_train_dir, f) for f in os.listdir(category_train_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    with torch.no_grad():
        for img_path in image_files:
            img = Image.open(img_path).convert('RGB')
            img_tensor = transform(img).unsqueeze(0).to(device)
            feature = model.forward_features(img_tensor)[:, 0]
            features.append(feature.cpu().numpy())

    if not features:
        return None, None

    features = np.vstack(features)
    
    pca = None
    if APPLY_PCA:
        n_components = min(PCA_COMPONENTS, features.shape[0], features.shape[1])
        pca = PCA(n_components=n_components)
        features = pca.fit_transform(features)
        
    return features, pca

# --- 8. Test Set Feature Extraction ---
def extract_test_features(model, category_test_dir, transform, pca=None):
    features = []
    image_paths = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for subdir, dirs, files in os.walk(category_test_dir):
        # Skip ground truth in test features
        if 'ground_truth' in subdir.split(os.path.sep):
            continue
            
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, f)
                image_paths.append(img_path)
                img = Image.open(img_path).convert('RGB')
                img_tensor = transform(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    feature = model.forward_features(img_tensor)[:, 0]
                    features.append(feature.cpu().numpy())

    if not features:
        return None, []

    features = np.vstack(features)
    if APPLY_PCA and pca is not None:
        features = pca.transform(features)
        
    return features, image_paths

# --- 9. Compute Anomaly Scores ---
def compute_anomaly_scores(memory_bank, test_features):
    memory_bank = torch.tensor(memory_bank, dtype=torch.float32)
    test_features = torch.tensor(test_features, dtype=torch.float32)
    memory_bank = F.normalize(memory_bank, p=2, dim=1)
    test_features = F.normalize(test_features, p=2, dim=1)
    similarity = torch.mm(test_features, memory_bank.T)
    anomaly_scores = 1 - torch.max(similarity, dim=1).values
    return anomaly_scores.numpy()

def visualize_results(category, image_paths, anomaly_scores, num_images=5):
    sorted_indices = np.argsort(anomaly_scores)[::-1]
    for i in range(min(num_images, len(sorted_indices))):
        idx = sorted_indices[i]
        img_path = image_paths[idx]
        score = anomaly_scores[idx]
        img = Image.open(img_path)
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title(f"{category} - Anomaly Score: {score:.4f}\n{os.path.basename(img_path)}")
        plt.axis('off')
        save_path = os.path.join(OUTPUT_DIR, f"{category}_top_{i}_score_{score:.4f}.png")
        plt.savefig(save_path)
        plt.close()

if __name__ == '__main__':
    train_transform = get_transforms(is_train=True)
    test_transform = get_transforms(is_train=False)
    dino_model = load_dino_model()

    categories = [os.path.join(DATASET_DIR, f) for f in os.listdir(DATASET_DIR) if os.path.isdir(os.path.join(DATASET_DIR, f))]
    
    for cat_path in categories:
        cat_name = os.path.basename(cat_path)
        train_dir = os.path.join(cat_path, 'train', 'good')
        test_dir = os.path.join(cat_path, 'test')
        
        if not os.path.exists(train_dir) or not os.path.exists(test_dir):
            print(f"Skipping {cat_name}: missing train or test directory.")
            continue
            
        print(f"\nProcessing category: {cat_name}")
        memory_bank, pca_model = build_feature_memory_bank(dino_model, train_dir, train_transform)
        if memory_bank is None:
            print(f"Skipping {cat_name}: no training features extracted.")
            continue

        test_features, test_image_paths = extract_test_features(dino_model, test_dir, test_transform, pca_model)
        if test_features is None:
            print(f"Skipping {cat_name}: no test features extracted.")
            continue

        scores = compute_anomaly_scores(memory_bank, test_features)
        
        avg_score = np.mean(scores)
        print(f"Category {cat_name} - Avg Anomaly Score: {avg_score:.4f}")
        
        visualize_results(cat_name, test_image_paths, scores)

    print("\nPipeline finished. Visualizations saved in 'outputs/'.")

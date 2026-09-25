
import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import joblib
import os
import time
from PIL import Image
from feature_extraction import FeatureExtractor, get_transforms
from config import CATEGORY_TYPES, HYBRID_WEIGHTS, ADAPTIVE_K

# --- Apple Minimalist Design CSS ---
st.set_page_config(page_title="Industrial Quality Control", layout="wide", page_icon="🔍")

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; color: #1d1d1f; }
    .main { background-color: #ffffff; }
    .stMetric { background-color: #f5f5f7; padding: 20px; border-radius: 18px; border: 1px solid #e5e5e7; }
    .app-header { font-size: 48px; font-weight: 600; letter-spacing: -0.02em; margin-bottom: 10px; text-align: center; }
    .app-subheader { font-size: 24px; font-weight: 400; color: #86868b; margin-bottom: 40px; text-align: center; }
    .status-pass { color: #2ecc71; font-weight: 600; font-size: 24px; }
    .status-fail { color: #e74c3c; font-weight: 600; font-size: 24px; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_extractor():
    return FeatureExtractor().to(torch.device('cpu'))

def load_category_assets(category):
    bank_path = f'memory_bank/{category}/bank.pt'
    pca_path = f'memory_bank/{category}/pca.pkl'
    cls_bank_path = f'memory_bank/{category}/cls_bank.pt'
    if not os.path.exists(bank_path): return None, None, None
    bank = torch.load(bank_path, map_location='cpu')
    pca = joblib.load(pca_path)
    cls_bank = torch.load(cls_bank_path, map_location='cpu')
    return bank, pca, cls_bank

def get_optimal_threshold(category):
    try:
        with open('outputs/evaluation_report.txt', 'r') as f:
            for line in f:
                if line.strip().startswith(category): return float(line.split()[-1])
    except: pass
    return 0.5

# --- Sidebar ---
st.sidebar.markdown("### 📊 Control Center")
selected_cat = st.sidebar.selectbox("Inspection Domain", sorted(list(CATEGORY_TYPES.keys())), index=0)
sensitivity = st.sidebar.slider("Sensitivity Bias", 0.5, 1.5, 1.0, 0.05, help="Increase to make the model stricter (detect smaller anomalies)")
threshold = get_optimal_threshold(selected_cat) / sensitivity

# --- Header ---
st.markdown('<div class="app-header">Quality Assurance. Redefined.</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-subheader">Intelligent inspection for {selected_cat.title()}</div>', unsafe_allow_html=True)

uploaded_files = st.file_uploader("Browse files to inspect", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

if uploaded_files:
    extractor = load_extractor()
    transform = get_transforms()
    bank, pca, cls_bank = load_category_assets(selected_cat)
    
    if bank is not None:
        bank = F.normalize(bank, p=2, dim=1)
        cls_bank = F.normalize(cls_bank, p=2, dim=1)
        
        for uploaded_file in uploaded_files:
            start_t = time.perf_counter()
            img = Image.open(uploaded_file).convert('RGB')
            img_tensor = transform(img).unsqueeze(0)
            
            with torch.no_grad():
                features = extractor(img_tensor)
                
                # Global
                cls_feat = F.normalize(features[0, 0:1, :], p=2, dim=1)
                cls_sim = torch.matmul(cls_feat, cls_bank.t())
                global_dist = np.sqrt(max(0.0, 2.0 - 2.0 * torch.max(cls_sim).item()))
                
                # Patch
                patch_feats = features[0, 1:, :].numpy()
                reduced = pca.transform(patch_feats)
                reduced = torch.from_numpy(reduced)
                reduced = F.normalize(reduced, p=2, dim=1)
                
                sim = torch.matmul(reduced, bank.t())
                max_sim, _ = torch.max(sim, dim=1)
                patch_dists = torch.sqrt(torch.clamp(2.0 - 2.0 * max_sim, min=0.0))
                
                # Smoothing
                dist_grid = patch_dists.view(1, 1, 28, 28)
                kernel = torch.ones((1, 1, 3, 3)) / 9.0
                dist_grid = F.conv2d(dist_grid, kernel, padding=1)
                
                # Aggregation
                cat_type = CATEGORY_TYPES.get(selected_cat, 'object')
                k = max(1, int(784 * ADAPTIVE_K[cat_type]))
                top_k_vals, _ = torch.topk(dist_grid.view(-1), k)
                patch_score = torch.mean(top_k_vals).item()
                final_score = 0.4 * global_dist + 0.6 * patch_score
                
                # --- FIXED Heatmap Normalization ---
                heatmap = dist_grid[0, 0].numpy()
                # Use a FIXED scale based on typical anomaly ranges (0.3 to 1.0)
                # This prevents "all red" on good images
                heatmap_vis = np.clip((heatmap - 0.3) / (0.7), 0, 1)
                heatmap_color = cv2.applyColorMap((heatmap_vis * 255).astype(np.uint8), cv2.COLORMAP_JET)
                
                img_cv = np.array(img.resize((224, 224)))
                overlay = cv2.addWeighted(img_cv, 0.7, cv2.resize(heatmap_color, (224, 224)), 0.3, 0)
            
            inf_ms = (time.perf_counter() - start_t) * 1000
            
            c1, c2, c3 = st.columns([1, 1, 1.5])
            with c1: st.image(img, caption="Source", use_container_width=True)
            with c2: st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), caption="Anomaly Heatmap", use_container_width=True)
            with c3:
                verdict = "ANOMALY" if final_score > threshold else "PASSED"
                v_class = "status-fail" if verdict == "ANOMALY" else "status-pass"
                st.markdown(f"**Verdict:** <span class='{v_class}'>{verdict}</span>", unsafe_allow_html=True)
                st.write(f"Latency: `{inf_ms:.1f} ms` | Score: `{final_score:.4f}`")
                st.progress(min(1.0, final_score / (threshold * 1.5)))
            st.markdown("---")
else:
    st.info("Ingest industrial images to start the inspection cycle.")

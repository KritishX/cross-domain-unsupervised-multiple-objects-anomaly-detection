
# --- Pipeline Configuration ---

# Model Settings
BACKBONE = 'vit_small_patch16_224.dino'
IMG_SIZE = 224
PATCH_GRID_SIZE = 14 # Original ViT output
UPSAMPLE_SIZE = 28   # Interpolated resolution

# Hybrid Feature Settings
PATCH_LAYERS = [11, 12] # Concat patches from these layers
CLS_LAYER = 12          # Use CLS from the final layer

# Memory Bank Settings
CORESET_SIZE = 5000
PCA_COMPONENTS = 256
DISTANCE_METRIC = 'euclidean_on_sphere'

# Scoring Settings
ADAPTIVE_K = {
    'texture': 0.15, # Top 15% for grid, carpet, etc.
    'object': 0.06   # Top 6% for discrete objects
}
HYBRID_WEIGHTS = {
    'global': 0.4,
    'patch': 0.6
}

# Category Metadata
CATEGORY_TYPES = {
    'grid': 'texture', 'carpet': 'texture', 'tile': 'texture', 'wood': 'texture',
    'bottle': 'object', 'capsule': 'object', 'pill': 'object', 'transistor': 'object',
    'metal_nut': 'object', 'screw': 'object', 'zipper': 'object', 'hazelnut': 'object',
    'cable': 'object', 'leather': 'object', 'toothbrush': 'object'
}

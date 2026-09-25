import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

# --- Configuration ---
OUTPUTS_DIR = 'outputs'
SCORES_FILE = os.path.join(OUTPUTS_DIR, 'anomaly_scores.csv')

def evaluate_performance():
    """Reads anomaly scores and computes per-category AUROC, F1, and optimal thresholds."""
    if not os.path.exists(SCORES_FILE):
        print(f"Error: {SCORES_FILE} not found. Run scoring first.")
        return

    # Load computed scores
    df = pd.read_csv(SCORES_FILE)
    categories = df['category'].unique()
    
    cat_metrics = []
    
    # Calculate performance for each category individually
    for cat in categories:
        cat_df = df[df['category'] == cat]
        y_true = cat_df['label'].values # 0 for good, 1 for anomaly
        y_scores = cat_df['score'].values
        
        # Calculate AUROC
        auroc = roc_auc_score(y_true, y_scores)
        
        # Determine the best F1-score and the threshold that produces it
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # Avoid division by zero
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_f1 = f1_scores[best_idx]
        # Thresholds from precision_recall_curve are one shorter than precision/recall
        best_threshold = thresholds[min(best_idx, len(thresholds)-1)]
        
        cat_metrics.append({
            'Category': cat,
            'AUROC': auroc,
            'Best F1': best_f1,
            'Optimal Threshold': best_threshold
        })

    # Create a summary table
    report_df = pd.DataFrame(cat_metrics).sort_values(by='AUROC', ascending=False)
    
    # --- Overall Reporting ---
    # Raw average across categories
    overall_auroc_raw = report_df['AUROC'].mean()
    
    # Normalized Overall AUROC Calculation
    # Step 1: Z-score normalization of scores within each category to align scales
    # This prevents categories with higher absolute scores from dominating the overall metric
    normalized_scores = []
    for cat in categories:
        cat_df = df[df['category'] == cat].copy()
        s = cat_df['score'].values
        # Z-normalize: (x - mean) / std
        normalized_s = (s - np.mean(s)) / (np.std(s) + 1e-8)
        cat_df['norm_score'] = normalized_s
        normalized_scores.append(cat_df)
    
    norm_df = pd.concat(normalized_scores)
    # Step 2: Compute AUROC using the pooled normalized scores
    overall_auroc_norm = roc_auc_score(norm_df['label'], norm_df['norm_score'])

    # Print Report to Console
    print("\n=== Anomaly Detection Evaluation Report ===\n")
    print(report_df.to_string(index=False))
    print("\n" + "="*43)
    print(f"\nOVERALL AUROC (Raw): {overall_auroc_raw:.4f}")
    print(f"OVERALL AUROC (Normalized): {overall_auroc_norm:.4f}")
    
    hard_cases = report_df[report_df['AUROC'] < 0.8]['Category'].tolist()
    if hard_cases:
        print(f"\nHard Cases (AUROC < 0.8): {', '.join(hard_cases)}")
    
    # Save Report to Disk
    with open(os.path.join(OUTPUTS_DIR, 'evaluation_report.txt'), 'w') as f:
        f.write("=== Anomaly Detection Evaluation Report ===\n\n")
        f.write(report_df.to_string(index=False))
        f.write(f"\n\nOVERALL AUROC (Raw): {overall_auroc_raw:.4f}")
        f.write(f"\nOVERALL AUROC (Normalized): {overall_auroc_norm:.4f}")
        if hard_cases:
            f.write(f"\n\nHard Cases: {', '.join(hard_cases)}")

    print(f"\nReport saved to {OUTPUTS_DIR}/evaluation_report.txt")

if __name__ == '__main__':
    evaluate_performance()
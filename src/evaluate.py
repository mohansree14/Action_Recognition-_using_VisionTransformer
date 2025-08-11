# evaluate.py
# Evaluation metrics: accuracy, confusion matrix, top-5 analysis
# 
# OUTPUT STRUCTURE:
# results/
# ├── {model_type}_evaluation/           # Individual model evaluation results
# │   ├── {model}_evaluation_results_{timestamp}.json
# │   ├── {model}_evaluation_{timestamp}.log
# │   └── {model}_confusion_matrix_{timestamp}.png
# ├── model_comparison/                  # Cross-model comparison results
# │   ├── model_comparison_results_{timestamp}.json
# │   ├── model_comparison_report_{timestamp}.txt
# │   ├── accuracy_comparison_{timestamp}.png
# │   ├── per_class_heatmap_{timestamp}.png
# │   └── performance_summary_{timestamp}.png
# └── {model_type}_model/                # Trained model checkpoints (from training)
#     ├── {model}_best_model_{timestamp}.pth
#     ├── {model}_training_config_{timestamp}.json
#     └── {model}_metrics_{timestamp}.json

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
import sys
import argparse
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model import get_timesformer_model, get_vit_model, get_videomae_model
from dataset import HMDBDataset
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, top_k_accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.preprocessing import label_binarize
import os
import json
import glob
from datetime import datetime

def ensure_results_directory():
    """Ensure the results directory structure exists"""
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Create subdirectories if they don't exist
    subdirs = ["timesformer_evaluation", "vit_evaluation", "videomae_evaluation", 
               "model_comparison", "timesformer_model", "vit_model", "videomae_model"]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(base_results_dir, subdir), exist_ok=True)
    
    print(f"Results directory structure verified: {os.path.abspath(base_results_dir)}")
    return base_results_dir

def find_best_model(model_type):
    """Find the best saved model for the given model type"""
    model_dir = os.path.join("results", f"{model_type}_model")
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"No saved models found for {model_type}. Train the model first.")
    
    # Find all model files for this type - collect ALL files then pick most recent
    model_patterns = [
        f"{model_type}_best_model_*.pth",  # Original pattern
        f"{model_type}_lr*_f*_s*_*.pth",   # New pattern from train.py
        f"{model_type}_*.pth"              # Any model file
    ]
    
    model_files = []
    for pattern in model_patterns:
        found_files = glob.glob(os.path.join(model_dir, pattern))
        model_files.extend(found_files)
        if found_files:
            print(f"Found {len(found_files)} model file(s) with pattern: {pattern}")
    
    # Remove duplicates
    model_files = list(set(model_files))
    
    if not model_files:
        raise FileNotFoundError(f"No trained model files found in {model_dir}")
    
    # Get the most recent model file
    latest_model = max(model_files, key=os.path.getctime)
    print(f"Using latest model: {os.path.basename(latest_model)}")
    
    # Extract timestamp from model file
    model_filename = os.path.basename(latest_model)
    
    # Try to extract timestamp - handle different naming patterns
    if '_best_model_' in model_filename:
        timestamp = model_filename.split('_')[-1].replace('.pth', '')
    else:
        # For new pattern: vit_lr000005_f8_s32_20250801_165823.pth
        parts = model_filename.replace('.pth', '').split('_')
        # Look for timestamp pattern (8 digits followed by 6 digits: YYYYMMDD_HHMMSS)
        timestamp_parts = []
        for i, part in enumerate(parts):
            if len(part) == 8 and part.isdigit():  # Date part YYYYMMDD
                if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():  # Time part HHMMSS
                    timestamp = f"{part}_{parts[i + 1]}"
                    break
        else:
            # Fallback: use last part
            timestamp = parts[-1] if parts else "unknown"
    
    # Try to find corresponding config and metrics files with the same timestamp
    config_patterns = [
        f"{model_type}_training_config_{timestamp}.json",
        f"{model_type}_*_config_{timestamp}.json",
        f"{model_type}_*config*.json"
    ]
    
    metrics_patterns = [
        f"{model_type}_metrics_{timestamp}.json",
        f"{model_type}_*_metrics_{timestamp}.json", 
        f"{model_type}_*metrics*.json"
    ]
    
    config_file = None
    metrics_file = None
    
    # Find config file
    for pattern in config_patterns:
        config_files = glob.glob(os.path.join(model_dir, pattern))
        if config_files:
            config_file = max(config_files, key=os.path.getctime)
            print(f"Found config file: {os.path.basename(config_file)}")
            break
    
    if not config_file:
        print(f"WARNING: No config file found, will use default configuration")
    
    # Find metrics file
    for pattern in metrics_patterns:
        metrics_files = glob.glob(os.path.join(model_dir, pattern))
        if metrics_files:
            metrics_file = max(metrics_files, key=os.path.getctime)
            print(f"Found metrics file: {os.path.basename(metrics_file)}")
            break
    
    if not metrics_file:
        print(f"WARNING: No metrics file found")
    
    return latest_model, config_file, metrics_file

def create_advanced_visualizations(all_labels, all_preds, all_logits, categories, model_type, eval_dir, timestamp):
    """Create improved visualizations with better PR curves and no ROC curves"""
    
    print(f"Creating advanced visualizations for {model_type.upper()}...")
    
    # Get unique labels present in test set
    unique_labels = np.unique(all_labels)
    present_categories = [categories[i] for i in unique_labels if i < len(categories)]
    n_classes = len(unique_labels)
    total_categories = len(categories)
    
    print(f"   Total categories in config: {total_categories}")
    print(f"   Categories present in test set: {n_classes}")
    print(f"   Present categories: {present_categories}")
    
    # Convert probabilities from logits using softmax
    from scipy.special import softmax
    all_probs = softmax(all_logits, axis=1)
    
    # Initialize paths dictionary
    viz_paths = {}
    
    # 1. Enhanced Precision-Recall Curves
    try:
        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import precision_recall_curve, average_precision_score
        
        print("   Creating enhanced PR curves...")
        
        plt.figure(figsize=(14, 10))
        
        # Use all categories for proper binarization
        all_class_labels = list(range(total_categories))
        y_bin = label_binarize(all_labels, classes=all_class_labels)
        
        # Handle case where we have fewer samples than classes
        if y_bin.shape[1] < total_categories:
            # Pad with zeros for missing classes
            y_bin_padded = np.zeros((len(all_labels), total_categories))
            for i, label in enumerate(all_labels):
                if label < total_categories:
                    y_bin_padded[i, label] = 1
            y_bin = y_bin_padded
        
        # Calculate AP scores for all present classes
        class_ap_scores = []
        
        for class_idx in range(min(total_categories, all_probs.shape[1])):
            if class_idx in unique_labels and np.sum(y_bin[:, class_idx]) > 0:
                y_true_binary = y_bin[:, class_idx]
                y_scores = all_probs[:, class_idx]
                
                # Calculate Average Precision
                ap_score = average_precision_score(y_true_binary, y_scores)
                class_ap_scores.append((class_idx, categories[class_idx], ap_score))
        
        # Sort by AP score (best performing classes first)
        class_ap_scores.sort(key=lambda x: x[2], reverse=True)
        
        # Select top performing classes for visualization (max 8 for clarity)
        top_classes = class_ap_scores[:min(8, len(class_ap_scores))]
        
        # Create distinct colors
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                 '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'][:len(top_classes)]
        
        macro_ap = 0
        plotted_classes = 0
        
        # Plot smooth PR curves for top performing classes
        for i, (class_idx, class_name, ap_score) in enumerate(top_classes):
            y_true_binary = y_bin[:, class_idx]
            y_scores = all_probs[:, class_idx]
            
            # Calculate PR curve
            precision, recall, thresholds = precision_recall_curve(y_true_binary, y_scores)
            
            # Smooth the curve by interpolation for better visualization
            if len(recall) > 3:
                from scipy import interpolate
                # Remove duplicate recall values for interpolation
                unique_recall, unique_indices = np.unique(recall, return_index=True)
                unique_precision = precision[unique_indices]
                
                if len(unique_recall) > 2:
                    # Interpolate to create smooth curve
                    f = interpolate.interp1d(unique_recall, unique_precision, 
                                           kind='linear', bounds_error=False, 
                                           fill_value=(unique_precision[0], unique_precision[-1]))
                    recall_smooth = np.linspace(0, 1, 200)
                    precision_smooth = f(recall_smooth)
                    precision_smooth = np.clip(precision_smooth, 0, 1)
                    
                    plt.plot(recall_smooth, precision_smooth, color=colors[i], 
                            linewidth=3, alpha=0.8,
                            label=f'{class_name[:20]} (AP={ap_score:.3f})')
                else:
                    plt.plot(recall, precision, color=colors[i], 
                            linewidth=3, alpha=0.8,
                            label=f'{class_name[:20]} (AP={ap_score:.3f})')
            else:
                plt.plot(recall, precision, color=colors[i], 
                        linewidth=3, alpha=0.8,
                        label=f'{class_name[:20]} (AP={ap_score:.3f})')
            
            macro_ap += ap_score
            plotted_classes += 1
        
        # Calculate macro average AP
        if plotted_classes > 0:
            macro_ap /= plotted_classes
        
        # Add baseline (random classifier line)
        baseline_precision = np.sum(all_labels >= 0) / len(all_labels)  # Proportion of positive samples
        plt.axhline(y=baseline_precision, color='gray', linestyle='--', 
                   alpha=0.7, linewidth=2, label=f'Random Baseline ({baseline_precision:.3f})')
        
        # Formatting
        plt.xlabel('Recall', fontsize=16, fontweight='bold')
        plt.ylabel('Precision', fontsize=16, fontweight='bold') 
        plt.title(f'{model_type.upper()} - Precision-Recall Curves\\n'
                 f'Top {plotted_classes} Classes (Macro mAP = {macro_ap:.3f})', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Better legend positioning
        plt.legend(loc='lower left', fontsize=11, frameon=True, fancybox=True, 
                  shadow=True, ncol=2 if len(top_classes) > 4 else 1)
        
        # Grid and formatting with better visual range
        plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
        
        # Set axis limits to focus on meaningful range and reduce empty space
        # Use dynamic limits based on data but ensure reasonable viewing area
        plt.xlim([0.0, 1.02])
        plt.ylim([max(0.0, baseline_precision - 0.1), 1.02])
        
        # Add performance indicators
        plt.axhline(y=0.5, color='red', linestyle=':', alpha=0.5, linewidth=1)
        plt.axhline(y=0.8, color='green', linestyle=':', alpha=0.5, linewidth=1)
        
        # Add summary statistics box
        total_samples = len(all_labels)
        textstr = f'Total Samples: {total_samples}\\n'
        textstr += f'Classes in Test: {n_classes}/{total_categories}\\n'
        textstr += f'Top {plotted_classes} shown'
        
        props = dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8)
        plt.text(0.98, 0.02, textstr, transform=plt.gca().transAxes, fontsize=12,
                verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        pr_curve_path = os.path.join(eval_dir, f'{model_type}_precision_recall_curves_{timestamp}.png')
        plt.tight_layout()
        plt.savefig(pr_curve_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        viz_paths['pr_curves'] = pr_curve_path
        
        print(f"   ✅ Enhanced PR curves saved: {os.path.basename(pr_curve_path)}")
        print(f"   📊 Showing top {plotted_classes} classes, Macro mAP: {macro_ap:.3f}")
        
    except Exception as e:
        print(f"   ⚠️ Error creating PR curves: {e}")
        viz_paths['pr_curves'] = None
    
    # 2. Individual Confidence Analysis Plots
    print("   Creating confidence analysis plots...")
    
    # Get prediction confidences
    max_probs = np.max(all_probs, axis=1)
    correct_mask = all_preds == all_labels
    
    # 3a. Confidence Distribution
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 25)
    
    plt.hist(max_probs[correct_mask], bins=bins, alpha=0.7, 
            label=f'Correct ({np.sum(correct_mask)} samples)', color='green', density=True)
    plt.hist(max_probs[~correct_mask], bins=bins, alpha=0.7, 
            label=f'Incorrect ({np.sum(~correct_mask)} samples)', color='red', density=True)
    
    plt.xlabel('Prediction Confidence', fontsize=14)
    plt.ylabel('Density', fontsize=14)
    plt.title(f'{model_type.upper()} - Confidence Distribution', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    confidence_dist_path = os.path.join(eval_dir, f'{model_type}_confidence_distribution_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(confidence_dist_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths['confidence_distribution'] = confidence_dist_path
    
    # 3b. Calibration Plot
    plt.figure(figsize=(10, 6))
    confidence_bins = np.linspace(0, 1, 11)
    bin_accuracies = []
    bin_centers = []
    bin_counts = []
    
    for i in range(len(confidence_bins)-1):
        bin_mask = (max_probs >= confidence_bins[i]) & (max_probs < confidence_bins[i+1])
        if np.sum(bin_mask) > 0:
            bin_accuracy = np.mean(correct_mask[bin_mask])
            bin_accuracies.append(bin_accuracy)
            bin_centers.append((confidence_bins[i] + confidence_bins[i+1]) / 2)
            bin_counts.append(np.sum(bin_mask))
    
    if bin_centers:
        plt.plot(bin_centers, bin_accuracies, 'o-', linewidth=3, markersize=8, color='blue')
        
        # Add sample count labels
        for x, y, count in zip(bin_centers, bin_accuracies, bin_counts):
            plt.annotate(f'n={count}', (x, y), textcoords="offset points", 
                        xytext=(0,15), ha='center', fontsize=10)
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.7, linewidth=2, label='Perfect Calibration')
    plt.xlabel('Confidence', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.title(f'{model_type.upper()} - Model Calibration', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    
    calibration_path = os.path.join(eval_dir, f'{model_type}_calibration_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(calibration_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths['calibration'] = calibration_path
    
    # 3c. Per-Class Confidence
    plt.figure(figsize=(16, 8))
    class_confidences = []
    class_names = []
    class_counts = []
    
    for i, label in enumerate(unique_labels):
        if i < len(present_categories):
            class_mask = all_labels == label
            if np.sum(class_mask) > 0:
                class_conf = max_probs[class_mask]
                class_confidences.append(class_conf)
                class_names.append(present_categories[i])
                class_counts.append(len(class_conf))
    
    if class_confidences:
        # Create box plot
        box_plot = plt.boxplot(class_confidences, labels=class_names, patch_artist=True)
        
        # Color boxes with better color scheme
        colors = plt.cm.Set2(np.linspace(0, 1, len(class_confidences)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.8)
            patch.set_linewidth(1.5)
        
        # Style whiskers, caps, and medians
        for whisker in box_plot['whiskers']:
            whisker.set_color('black')
            whisker.set_linewidth(1.5)
        for cap in box_plot['caps']:
            cap.set_color('black')
            cap.set_linewidth(2)
        for median in box_plot['medians']:
            median.set_color('darkred')
            median.set_linewidth(2.5)
        
        # Add sample count labels with better positioning
        for i, (name, count) in enumerate(zip(class_names, class_counts)):
            plt.text(i+1, 0.02, f'n={count}', ha='center', va='bottom', fontsize=11, 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        plt.ylabel('Prediction Confidence', fontsize=16, fontweight='bold')
        plt.title(f'{model_type.upper()} - Per-Class Confidence Distribution', 
                 fontsize=18, fontweight='bold', pad=20)
        
        # Improve x-axis labels
        plt.xticks(rotation=45, ha='right', fontsize=12, fontweight='bold')
        
        # Clean grid and axis limits
        plt.grid(True, alpha=0.4, axis='y', linestyle='-', linewidth=0.5)
        plt.ylim(0.0, 1.0)  # Tight y-axis limits to reduce empty space
        
        # Add horizontal reference lines
        plt.axhline(y=0.5, color='gray', linestyle=':', alpha=0.6, linewidth=1.5, label='Random Threshold')
        plt.axhline(y=0.8, color='orange', linestyle=':', alpha=0.6, linewidth=1.5, label='High Confidence')
        
        # Add subtle legend
        plt.legend(loc='upper right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    perclass_conf_path = os.path.join(eval_dir, f'{model_type}_perclass_confidence_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(perclass_conf_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths['perclass_confidence'] = perclass_conf_path
    
    # 4. Enhanced Confusion Matrix
    print("   Creating enhanced confusion matrix...")
    
    plt.figure(figsize=(max(12, n_classes * 0.8), max(10, n_classes * 0.7)))
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)  # Avoid division by zero
    
    # Create custom annotation matrix with better formatting
    annot_matrix = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            pct = cm_normalized[i, j] * 100
            if count == 0:
                annot_matrix[i, j] = '0'
            else:
                annot_matrix[i, j] = f'{count}\n({pct:.1f}%)'
    
    # Create heatmap
    sns.heatmap(cm_normalized, 
                annot=annot_matrix, 
                fmt='',
                cmap='Blues',
                xticklabels=present_categories, 
                yticklabels=present_categories,
                cbar_kws={'label': 'Normalized Frequency'},
                square=True,
                linewidths=0.5,
                annot_kws={'fontsize': max(8, 14 - n_classes)})
    
    plt.title(f'{model_type.upper()} - Detailed Confusion Matrix\n(Values: Count and Percentage)', 
             fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Predicted Label', fontsize=14, labelpad=10)
    plt.ylabel('True Label', fontsize=14, labelpad=10)
    
    # Rotate labels for better readability
    plt.xticks(rotation=45, ha='right', fontsize=max(8, 12 - n_classes//2))
    plt.yticks(rotation=0, fontsize=max(8, 12 - n_classes//2))
    
    detailed_cm_path = os.path.join(eval_dir, f'{model_type}_detailed_confusion_matrix_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(detailed_cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths['detailed_confusion_matrix'] = detailed_cm_path
    
    # 5. Top-k Accuracy Analysis
    print("   Creating top-k accuracy analysis...")
    
    plt.figure(figsize=(12, 8))
    
    k_values = list(range(1, min(6, n_classes + 1)))  # Adapt to number of classes
    topk_accuracies = []
    
    for k in k_values:
        # Get top k predictions
        topk_preds = np.argsort(all_logits, axis=1)[:, -k:]
        topk_acc = np.mean([label in preds for label, preds in zip(all_labels, topk_preds)])
        topk_accuracies.append(topk_acc * 100)
    
    # Create more professional plot with gradient fill
    plt.plot(k_values, topk_accuracies, 'o-', linewidth=4, markersize=12, 
             color='purple', markerfacecolor='darkviolet', markeredgecolor='white', 
             markeredgewidth=2, alpha=0.9)
    
    # Add gradient fill under the curve
    plt.fill_between(k_values, topk_accuracies, alpha=0.3, color='purple')
    
    # Add value labels on points with better styling
    for k, acc in zip(k_values, topk_accuracies):
        plt.annotate(f'{acc:.1f}%', (k, acc), textcoords="offset points", 
                    xytext=(0, 20), ha='center', fontweight='bold', fontsize=14,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                             edgecolor='purple', alpha=0.9))
    
    plt.xlabel('k (Top-k)', fontsize=16, fontweight='bold')
    plt.ylabel('Top-k Accuracy (%)', fontsize=16, fontweight='bold')
    plt.title(f'{model_type.upper()} - Top-k Accuracy Analysis', fontsize=18, fontweight='bold', pad=20)
    
    # Professional grid styling
    plt.grid(True, alpha=0.4, linestyle='-', linewidth=0.5)
    plt.xticks(k_values, fontsize=14, fontweight='bold')
    plt.yticks(fontsize=14)
    
    # Dynamic y-axis limits to reduce empty space and focus on meaningful range
    min_acc = min(topk_accuracies)
    max_acc = max(topk_accuracies)
    y_margin = max(2, (max_acc - min_acc) * 0.1)  # At least 2% margin
    
    # Set professional y-axis range
    y_min = max(0, min_acc - y_margin - 5)  # Start a bit below minimum
    y_max = min(100, max_acc + y_margin + 2)  # End a bit above maximum
    
    plt.ylim(y_min, y_max)
    
    # Add performance reference lines
    if min_acc > 50:
        plt.axhline(y=50, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Baseline (50%)')
    if max_acc < 95:
        plt.axhline(y=95, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent (95%)')
    
    # Add subtle legend if reference lines exist
    if min_acc > 50 or max_acc < 95:
        plt.legend(loc='lower right', fontsize=12, frameon=True, fancybox=True, shadow=True)
    
    topk_path = os.path.join(eval_dir, f'{model_type}_topk_analysis_{timestamp}.png')
    plt.tight_layout()
    plt.savefig(topk_path, dpi=300, bbox_inches='tight')
    plt.close()
    viz_paths['topk_analysis'] = topk_path
    
    print(f"   Advanced visualizations completed:")
    for name, path in viz_paths.items():
        if path:
            print(f"      {name}: {os.path.basename(path)}")
    
    return viz_paths

def load_model_checkpoint(model_path, model_type, num_classes, device):
    """Load trained model from checkpoint"""
    # Load checkpoint first to get actual number of classes
    checkpoint = torch.load(model_path, map_location=device)
    
    # Try to determine num_classes from checkpoint
    actual_num_classes = num_classes
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        # Check classifier layer to get actual number of classes
        if 'classifier.weight' in state_dict:
            actual_num_classes = state_dict['classifier.weight'].shape[0]
        elif 'head.weight' in state_dict:
            actual_num_classes = state_dict['head.weight'].shape[0]
    
    print(f"Checkpoint trained with {actual_num_classes} classes, current config has {num_classes} classes")
    
    # Initialize model architecture with actual number of classes from checkpoint
    if model_type == "timesformer":
        model = get_timesformer_model(num_classes=actual_num_classes)
    elif model_type == "vit":
        model = get_vit_model(num_classes=actual_num_classes)
    elif model_type == "videomae":
        model = get_videomae_model(num_classes=actual_num_classes)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load checkpoint
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Loaded trained model from: {model_path}")
    print(f"   Best validation accuracy: {checkpoint['val_accuracy']:.3f}")
    print(f"   Trained at epoch: {checkpoint['epoch']}")
    
    return model, checkpoint

def evaluate_model(model_type="timesformer", use_processed=True, model_path=None):
    # Ensure results directory structure exists
    ensure_results_directory()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Find and load the best trained model, or use provided path
    config_path = None
    metrics_path = None
    if model_path is None:
        try:
            model_path, config_path, metrics_path = find_best_model(model_type)
            print(f"Found model files for {model_type.upper()}")
        except FileNotFoundError as e:
            print(f"ERROR: {e}")
            print(f"Please train the {model_type} model first using: python train.py {model_type}")
            return
    else:
        if not os.path.exists(model_path):
            print(f"ERROR: Provided model path does not exist: {model_path}")
            return
        print(f"Using provided model checkpoint: {model_path}")
        # Try to find config and metrics in the same directory as the model
        model_dir = os.path.dirname(model_path)
        config_candidates = [f for f in os.listdir(model_dir) if f.endswith('.json') and 'config' in f]
        metrics_candidates = [f for f in os.listdir(model_dir) if f.endswith('.json') and 'metrics' in f]
        if config_candidates:
            config_path = os.path.join(model_dir, config_candidates[0])
        if metrics_candidates:
            metrics_path = os.path.join(model_dir, metrics_candidates[0])
    
    # Load configuration
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            training_config = json.load(f)
        print(f"Loaded training configuration")
    else:
        print(f"WARNING: No training config found, using default configuration")
        training_config = None
    
    with open("configs/coursework_config.yaml") as f:
        config = yaml.safe_load(f)
    
    categories = config['dataset']['categories']
    
    # Use same dataset path logic as training
    original_data_path = config['dataset']['root_dir']
    processed_data_path = os.path.join("results", "HMDB_simp_processed")
    
    # If we have training config, check if the training dataset path exists locally
    if training_config and 'dataset_path' in training_config:
        training_dataset_path = training_config['dataset_path']
        if os.path.exists(training_dataset_path):
            dataset_path = training_dataset_path
            print(f"USING TRAINING DATASET PATH: {dataset_path}")
        else:
            print(f"Training dataset path not found: {training_dataset_path}")
            # Fall back to local processed or original data
            if use_processed and os.path.exists(processed_data_path):
                dataset_path = processed_data_path
                print(f"FALLING BACK TO PROCESSED DATASET: {os.path.abspath(dataset_path)}")
            else:
                dataset_path = original_data_path
                print(f"FALLING BACK TO ORIGINAL DATASET: {dataset_path}")
    elif use_processed and os.path.exists(processed_data_path):
        dataset_path = processed_data_path
        print(f"USING PROCESSED DATASET: {os.path.abspath(dataset_path)}")
    else:
        dataset_path = original_data_path
        print(f"USING ORIGINAL DATASET: {dataset_path}")
    
    # Load trained model
    model, checkpoint = load_model_checkpoint(model_path, model_type, len(categories), device)
    
    # Create test dataset (using test split for true evaluation - prevents overfitting assessment)
    # Determine if we're using processed data based on the dataset path
    using_processed_data = 'processed' in dataset_path.lower()
    test_dataset = HMDBDataset(root_dir=dataset_path, categories=categories, mode='test', use_processed=using_processed_data)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, pin_memory=True)
    
    print(f"Dataset path: {dataset_path}")
    print(f"Using processed data: {using_processed_data}")
    print(f"Evaluating on {len(test_dataset)} test samples")
    
    # Load feature extractor
    extractor = AutoFeatureExtractor.from_pretrained(config['model']['model_name'])
    
    # Evaluation
    print("Running evaluation...")
    all_preds = []
    all_labels = []
    all_logits = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            # Move data to device
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Convert tensor to numpy for feature extractor
            videos_numpy = []
            pixel_values_cpu = pixel_values.cpu()
            for video in pixel_values_cpu:
                video_frames = []
                for frame in video:
                    frame_np = frame.permute(1, 2, 0).numpy()
                    frame_np = (frame_np * 255).astype('uint8')
                    video_frames.append(frame_np)
                videos_numpy.append(video_frames)
            
            # Process with feature extractor
            inputs = extractor(videos_numpy, return_tensors="pt")
            inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Forward pass
            outputs = model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=1)
            
            # Store results
            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Store logits properly to maintain 2D shape
            batch_logits = outputs.logits.cpu().numpy()
            if len(all_logits) == 0:
                all_logits = batch_logits
            else:
                all_logits = np.vstack([all_logits, batch_logits])
            
            if batch_idx % 10 == 0:
                print(f"   Processed batch {batch_idx}/{len(test_loader)}")
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Calculate metrics
    accuracy = np.mean(all_preds == all_labels)
    
    # Calculate top-5 accuracy (efficient vectorized version)
    try:
        from sklearn.metrics import top_k_accuracy_score
        
        # Check if all_logits is properly formed
        if len(all_logits) == 0:
            print("   Warning: No logits collected, skipping advanced metrics")
            top5_accuracy = 0.0
        else:
            # Ensure all_logits has proper 2D shape
            if all_logits.ndim == 1 and len(all_logits) > 0:
                # If it's 1D, try to reshape or skip
                print(f"   Warning: all_logits shape is {all_logits.shape}, skipping top-k accuracy")
                top5_accuracy = 0.0
            elif all_logits.ndim == 2:
                # Get unique classes in test data
                unique_classes = np.unique(all_labels)
                n_test_classes = len(unique_classes)
                n_model_classes = all_logits.shape[1]
                
                print(f"   Test data classes: {n_test_classes}, Model classes: {n_model_classes}")
                
                # If we have class mismatch, we need to align the logits
                if n_test_classes != n_model_classes:
                    print(f"   Adjusting for class mismatch...")
                    # Create a mapping from test classes to model classes
                    aligned_logits = np.zeros((len(all_labels), n_test_classes))
                    for i, class_idx in enumerate(unique_classes):
                        if class_idx < n_model_classes:
                            aligned_logits[:, i] = all_logits[:, class_idx]
                    
                    # Remap labels to align with the reduced logits
                    label_mapping = {old_label: new_label for new_label, old_label in enumerate(unique_classes)}
                    aligned_labels = np.array([label_mapping[label] for label in all_labels])
                    
                    top5_accuracy = top_k_accuracy_score(aligned_labels, aligned_logits, k=min(5, n_test_classes))
                else:
                    top5_accuracy = top_k_accuracy_score(all_labels, all_logits, k=5)
            else:
                print(f"   Warning: Unexpected all_logits shape {all_logits.shape}")
                top5_accuracy = 0.0
            
    except (ImportError, ValueError) as e:
        print(f"   Sklearn top-k error: {e}")
        # Fallback to manual calculation (vectorized)
        top5_preds = np.argsort(all_logits, axis=1)[:, -5:]  # Top 5 predictions for all samples
        top5_accuracy = np.mean([label in preds for label, preds in zip(all_labels, top5_preds)])
    
    # Calculate advanced metrics
    try:
        from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, cohen_kappa_score
        
        # Check if we have valid data for metrics calculation
        if len(all_labels) == 0 or len(all_preds) == 0:
            print("   Warning: Empty predictions or labels, skipping advanced metrics")
            advanced_metrics = {}
        elif len(np.unique(all_labels)) < 2:
            print("   Warning: Need at least 2 classes for advanced metrics")
            advanced_metrics = {}
        else:
            # Overall precision, recall, f1 with different averaging strategies
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='macro', zero_division=0, warn_for=()
            )
            precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='micro', zero_division=0, warn_for=()
            )
            precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
                all_labels, all_preds, average='weighted', zero_division=0, warn_for=()
            )
            
            # Balanced accuracy (good for imbalanced datasets)
            balanced_acc = balanced_accuracy_score(all_labels, all_preds)
            
            # Cohen's Kappa (agreement measure)
            kappa = cohen_kappa_score(all_labels, all_preds)
            
            advanced_metrics = {
                'precision_macro': float(precision_macro),
                'recall_macro': float(recall_macro),
                'f1_macro': float(f1_macro),
                'precision_micro': float(precision_micro),
                'recall_micro': float(recall_micro),
                'f1_micro': float(f1_micro),
                'precision_weighted': float(precision_weighted),
                'recall_weighted': float(recall_weighted),
                'f1_weighted': float(f1_weighted),
                'balanced_accuracy': float(balanced_acc),
                'cohens_kappa': float(kappa)
            }
            
            print(f"   Balanced Accuracy: {balanced_acc*100:.2f}%")
            print(f"   F1-Score (Macro): {f1_macro:.3f}")
            print(f"   Cohen's Kappa: {kappa:.3f}")
        
    except ImportError:
        print("   Advanced metrics require scikit-learn")
        advanced_metrics = {}
    except Exception as e:
        print(f"   Error calculating advanced metrics: {e}")
        advanced_metrics = {}
    
    print(f"\nEvaluation Results for {model_type.upper()}:")
    print(f"   Top-1 Accuracy: {accuracy*100:.2f}%")
    print(f"   Top-5 Accuracy: {top5_accuracy*100:.2f}%")
    
    # Create evaluation directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_dir = os.path.join("results", f"{model_type}_evaluation")
    os.makedirs(eval_dir, exist_ok=True)
    
    # Check if we have any samples to evaluate
    if len(all_labels) == 0 or len(all_preds) == 0:
        print("ERROR: No samples to evaluate. Check dataset path and data availability.")
        return None
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Get unique labels and corresponding category names
    unique_labels = np.unique(all_labels)
    present_categories = [categories[i] for i in unique_labels if i < len(categories)]
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=present_categories, yticklabels=present_categories, cbar=True)
    plt.title(f'{model_type.upper()} - Confusion Matrix\nAccuracy: {accuracy*100:.2f}% ({len(present_categories)} classes)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = os.path.join(eval_dir, f'{model_type}_confusion_matrix_{timestamp}.png')
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create advanced visualizations
    try:
        viz_paths = create_advanced_visualizations(
            all_labels, all_preds, all_logits, categories, model_type, eval_dir, timestamp
        )
        print(f"   Advanced visualizations created successfully")
    except Exception as e:
        print(f"   Warning: Could not create advanced visualizations: {e}")
        viz_paths = {}
    
    # Generate classification report
    try:
        from sklearn.metrics import classification_report
        
        # Get unique labels present in the test set
        unique_labels = np.unique(all_labels)
        present_categories = [categories[i] for i in unique_labels if i < len(categories)]
        
        print(f"Classes found in test set: {len(unique_labels)} out of {len(categories)} total categories")
        print(f"   Present categories: {present_categories}")
        
        # Generate classification report only for present categories
        class_report = classification_report(all_labels, all_preds, 
                                           labels=unique_labels,
                                           target_names=present_categories, 
                                           output_dict=True)
        class_report_text = classification_report(all_labels, all_preds, 
                                                 labels=unique_labels,
                                                 target_names=present_categories)
    except ImportError:
        print("WARNING: scikit-learn not available. Skipping detailed classification report.")
        class_report = None
        class_report_text = "Classification report requires scikit-learn"
    except Exception as e:
        print(f"WARNING: Error generating classification report: {e}")
        class_report = None
        class_report_text = f"Classification report error: {e}"
    
    # Per-class accuracy (only for classes present in test set)
    unique_labels = np.unique(all_labels)
    per_class_acc = {}
    class_distribution = {}
    
    for label in unique_labels:
        if label < len(categories):
            category = categories[label]
            class_mask = all_labels == label
            if np.sum(class_mask) > 0:
                class_count = np.sum(class_mask)
                class_acc = np.mean(all_preds[class_mask] == all_labels[class_mask])
                per_class_acc[category] = class_acc
                class_distribution[category] = int(class_count)
    
    # Analyze class imbalance impact
    if class_distribution:
        min_samples = min(class_distribution.values())
        max_samples = max(class_distribution.values())
        imbalance_ratio = max_samples / min_samples if min_samples > 0 else float('inf')
        
        print(f"   Class Distribution - Min: {min_samples}, Max: {max_samples}, Ratio: {imbalance_ratio:.2f}")
        
        # Correlation between class size and accuracy
        class_sizes = [class_distribution.get(cat, 0) for cat in per_class_acc.keys()]
        class_accs = list(per_class_acc.values())
        
        if len(class_sizes) > 1:
            correlation = np.corrcoef(class_sizes, class_accs)[0, 1]
            print(f"   Size-Accuracy Correlation: {correlation:.3f}")
    
    # Save detailed evaluation results
    unique_labels = np.unique(all_labels)
    present_categories = [categories[i] for i in unique_labels if i < len(categories)]
    
    evaluation_results = {
        "model_type": model_type,
        "model_path": model_path,
        "evaluation_timestamp": timestamp,
        "dataset_info": {
            "dataset_path": dataset_path,
            "num_samples": len(test_dataset),
            "num_categories_total": len(categories),
            "num_categories_present": len(present_categories),
            "all_categories": categories,
            "present_categories": present_categories,
            "present_labels": unique_labels.tolist(),
            "class_distribution": class_distribution,
            "imbalance_ratio": imbalance_ratio if 'imbalance_ratio' in locals() else 1.0
        },
        "metrics": {
            "top1_accuracy": float(accuracy),
            "top5_accuracy": float(top5_accuracy),
            "per_class_accuracy": per_class_acc,
            **advanced_metrics  # Include all advanced metrics
        },
        "training_info": {
            "best_epoch": checkpoint['epoch'],
            "training_val_accuracy": checkpoint['val_accuracy'],
            "training_val_loss": checkpoint['val_loss']
        } if checkpoint else {
            "best_epoch": "Unknown",
            "training_val_accuracy": "Unknown", 
            "training_val_loss": "Unknown"
        },
        "confusion_matrix": cm.tolist(),
        "classification_report": class_report,
        "visualization_paths": {
            "confusion_matrix": cm_path,
            **viz_paths  # Include advanced visualization paths
        }
    }
    
    # Save evaluation results
    results_path = os.path.join(eval_dir, f'{model_type}_evaluation_results_{timestamp}.json')
    with open(results_path, 'w') as f:
        json.dump(evaluation_results, f, indent=2)
    
    # Save detailed log
    log_path = os.path.join(eval_dir, f'{model_type}_evaluation_{timestamp}.log')
    with open(log_path, 'w') as f:
        f.write(f"Model Evaluation Report - {model_type.upper()}\n")
        f.write("="*50 + "\n\n")
        f.write(f"Model Path: {model_path}\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Dataset: {dataset_path}\n")
        f.write(f"Test Samples: {len(test_dataset)}\n\n")
        
        f.write("METRICS:\n")
        f.write(f"Top-1 Accuracy: {accuracy*100:.2f}%\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy*100:.2f}%\n\n")
        
        f.write("ADVANCED METRICS:\n")
        if 'advanced_metrics' in locals() and advanced_metrics:
            f.write(f"Balanced Accuracy: {advanced_metrics.get('balanced_accuracy', 0)*100:.2f}%\n")
            f.write(f"F1-Score (Macro): {advanced_metrics.get('f1_macro', 0):.3f}\n")
            f.write(f"F1-Score (Weighted): {advanced_metrics.get('f1_weighted', 0):.3f}\n")
            f.write(f"Precision (Macro): {advanced_metrics.get('precision_macro', 0):.3f}\n")
            f.write(f"Recall (Macro): {advanced_metrics.get('recall_macro', 0):.3f}\n")
            f.write(f"Cohen's Kappa: {advanced_metrics.get('cohens_kappa', 0):.3f}\n")
        f.write("\n")
        
        f.write("CLASS DISTRIBUTION:\n")
        if 'class_distribution' in locals():
            for category, count in sorted(class_distribution.items()):
                acc = per_class_acc.get(category, 0) * 100
                f.write(f"{category}: {count} samples ({acc:.2f}% accuracy)\n")
        f.write("\n")
        
        f.write("PER-CLASS ACCURACY:\n")
        for category, acc in per_class_acc.items():
            f.write(f"{category}: {acc*100:.2f}%\n")
        f.write("\n")
        
        f.write("CLASSIFICATION REPORT:\n")
        f.write(class_report_text)
        f.write("\n\nCONFUSION MATRIX:\n")
        f.write(str(cm))
    
    # Print summary
    print(f"\nEvaluation completed! Files saved in: {os.path.abspath(eval_dir)}")
    print(f"   Results: {os.path.basename(results_path)}")
    print(f"   Log: {os.path.basename(log_path)}")
    print(f"   Confusion Matrix: {os.path.basename(cm_path)}")
    print(f"   Full directory: {os.path.abspath(eval_dir)}")
    
    # Show top and bottom performing classes
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1], reverse=True)
    print(f"\nBest performing classes:")
    for category, acc in sorted_classes[:3]:
        print(f"   {category}: {acc*100:.2f}%")
    
    print(f"\nWorst performing classes:")
    for category, acc in sorted_classes[-3:]:
        print(f"   {category}: {acc*100:.2f}%")
    
    return evaluation_results

def evaluate_all_models(use_processed=True):
    """Evaluate all three models and create comparison visualizations"""
    # Ensure results directory structure exists
    ensure_results_directory()
    
    print("Starting comprehensive evaluation of all models")
    print("="*70)
    
    model_types = ["timesformer", "vit", "videomae"]
    all_results = {}
    successful_evaluations = []
    
    # Evaluate each model
    for model_type in model_types:
        print(f"\nEvaluating {model_type.upper()} model...")
        print("-" * 50)
        
        try:
            results = evaluate_model(model_type, use_processed)
            if results:
                all_results[model_type] = results
                successful_evaluations.append(model_type)
                print(f"{model_type.upper()} evaluation completed!")
        except Exception as e:
            print(f"ERROR: {model_type.upper()} evaluation failed: {str(e)}")
            print(f"Make sure to train the {model_type} model first")
            continue
    
    if not successful_evaluations:
        print("ERROR: No models could be evaluated. Please train the models first.")
        return
    
    # Create comparison visualizations
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join("results", "model_comparison")
    os.makedirs(comparison_dir, exist_ok=True)
    
    print(f"\nCreating model comparison visualizations...")
    
    # 1. Accuracy Comparison Bar Chart
    models = list(all_results.keys())
    top1_accuracies = [all_results[model]['metrics']['top1_accuracy'] * 100 for model in models]
    top5_accuracies = [all_results[model]['metrics']['top5_accuracy'] * 100 for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Top-1 Accuracy Comparison
    bars1 = ax1.bar(models, top1_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_title('Top-1 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars1, top1_accuracies):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    # Top-5 Accuracy Comparison
    bars2 = ax2.bar(models, top5_accuracies, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_title('Top-5 Accuracy Comparison', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_ylim(0, 100)
    
    # Add value labels on bars
    for bar, acc in zip(bars2, top5_accuracies):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    accuracy_comparison_path = os.path.join(comparison_dir, f'accuracy_comparison_{timestamp}.png')
    plt.savefig(accuracy_comparison_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Per-Class Accuracy Heatmap
    categories = list(all_results[models[0]]['dataset_info']['all_categories'])
    per_class_data = []
    
    for model in models:
        model_per_class = []
        per_class_acc = all_results[model]['metrics']['per_class_accuracy']
        for category in categories:
            model_per_class.append(per_class_acc.get(category, 0) * 100)
        per_class_data.append(model_per_class)
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(per_class_data, 
                xticklabels=categories, 
                yticklabels=[m.upper() for m in models],
                annot=True, 
                fmt='.1f', 
                cmap='RdYlGn', 
                cbar_kws={'label': 'Accuracy (%)'})
    plt.title('Per-Class Accuracy Comparison Across Models', fontsize=16, fontweight='bold')
    plt.xlabel('Action Categories', fontsize=12)
    plt.ylabel('Models', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    heatmap_path = os.path.join(comparison_dir, f'per_class_heatmap_{timestamp}.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Model Performance Summary Table
    plt.figure(figsize=(12, 6))
    
    summary_data = []
    for model in models:
        result = all_results[model]
        summary_data.append([
            model.upper(),
            f"{result['metrics']['top1_accuracy']*100:.2f}%",
            f"{result['metrics']['top5_accuracy']*100:.2f}%",
            f"{result['training_info']['best_epoch']}",
            f"{result['training_info']['training_val_accuracy']:.3f}",
            f"{result['dataset_info']['num_samples']}"
        ])
    
    table = plt.table(cellText=summary_data,
                     colLabels=['Model', 'Top-1 Acc', 'Top-5 Acc', 'Best Epoch', 'Train Val Acc', 'Test Samples'],
                     cellLoc='center',
                     loc='center',
                     colColours=['lightblue']*6)
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 2)
    
    plt.axis('off')
    plt.title('Model Performance Summary', fontsize=16, fontweight='bold', pad=20)
    
    summary_table_path = os.path.join(comparison_dir, f'performance_summary_{timestamp}.png')
    plt.savefig(summary_table_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create advanced comparison visualizations
    try:
        advanced_viz_paths = create_model_comparison_visualizations(all_results, comparison_dir, timestamp)
        print(f"   Advanced comparison visualizations created")
    except Exception as e:
        print(f"   Warning: Could not create advanced comparison visualizations: {e}")
        advanced_viz_paths = {}
    
    # 4. Save comprehensive comparison results
    comparison_results = {
        "comparison_timestamp": timestamp,
        "models_evaluated": models,
        "dataset_info": all_results[models[0]]['dataset_info'],
        "comparison_metrics": {
            "best_top1_model": max(models, key=lambda m: all_results[m]['metrics']['top1_accuracy']),
            "best_top5_model": max(models, key=lambda m: all_results[m]['metrics']['top5_accuracy']),
            "accuracy_rankings": {
                "top1": sorted(models, key=lambda m: all_results[m]['metrics']['top1_accuracy'], reverse=True),
                "top5": sorted(models, key=lambda m: all_results[m]['metrics']['top5_accuracy'], reverse=True)
            }
        },
        "individual_results": all_results
    }
    
    comparison_results_path = os.path.join(comparison_dir, f'model_comparison_results_{timestamp}.json')
    with open(comparison_results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    # Create detailed comparison report
    report_path = os.path.join(comparison_dir, f'model_comparison_report_{timestamp}.txt')
    with open(report_path, 'w') as f:
        f.write("MODEL COMPARISON REPORT\n")
        f.write("="*50 + "\n\n")
        f.write(f"Evaluation Date: {timestamp}\n")
        f.write(f"Models Evaluated: {', '.join([m.upper() for m in models])}\n\n")
        
        f.write("ACCURACY SUMMARY:\n")
        f.write("-"*30 + "\n")
        for model in models:
            result = all_results[model]
            f.write(f"{model.upper()}:\n")
            f.write(f"  Top-1 Accuracy: {result['metrics']['top1_accuracy']*100:.2f}%\n")
            f.write(f"  Top-5 Accuracy: {result['metrics']['top5_accuracy']*100:.2f}%\n")
            f.write(f"  Training Epochs: {result['training_info']['best_epoch']}\n")
            f.write(f"  Training Val Acc: {result['training_info']['training_val_accuracy']:.3f}\n\n")
        
        f.write("RANKINGS:\n")
        f.write("-"*30 + "\n")
        f.write("Top-1 Accuracy Ranking:\n")
        for i, model in enumerate(comparison_results['comparison_metrics']['accuracy_rankings']['top1'], 1):
            acc = all_results[model]['metrics']['top1_accuracy'] * 100
            f.write(f"  {i}. {model.upper()}: {acc:.2f}%\n")
        
        f.write("\nTop-5 Accuracy Ranking:\n")
        for i, model in enumerate(comparison_results['comparison_metrics']['accuracy_rankings']['top5'], 1):
            acc = all_results[model]['metrics']['top5_accuracy'] * 100
            f.write(f"  {i}. {model.upper()}: {acc:.2f}%\n")
    
    # Print final summary
    print(f"\nComprehensive evaluation completed!")
    print(f"Models evaluated: {', '.join([m.upper() for m in models])}")
    print(f"\nRESULTS SUMMARY:")
    print("-" * 40)
    
    for model in models:
        result = all_results[model]
        print(f"{model.upper():>12}: Top-1: {result['metrics']['top1_accuracy']*100:5.2f}% | Top-5: {result['metrics']['top5_accuracy']*100:5.2f}%")
    
    best_model = comparison_results['comparison_metrics']['best_top1_model']
    best_acc = all_results[best_model]['metrics']['top1_accuracy'] * 100
    print(f"\nBest performing model: {best_model.upper()} ({best_acc:.2f}%)")
    
    print(f"\nComparison files saved in: {os.path.abspath(comparison_dir)}")
    print(f"   Results: {os.path.basename(comparison_results_path)}")
    print(f"   Report: {os.path.basename(report_path)}")
    print(f"   Accuracy Chart: {os.path.basename(accuracy_comparison_path)}")
    print(f"   Heatmap: {os.path.basename(heatmap_path)}")
    print(f"   Summary Table: {os.path.basename(summary_table_path)}")
    if advanced_viz_paths:
        for viz_name, viz_path in advanced_viz_paths.items():
            if viz_path:
                print(f"   {viz_name.replace('_', ' ').title()}: {os.path.basename(viz_path)}")
    print(f"   Full directory: {os.path.abspath(comparison_dir)}")
    
    return comparison_results

def create_model_comparison_visualizations(all_results, comparison_dir, timestamp):
    """Create advanced model comparison visualizations"""
    
    models = list(all_results.keys())
    n_models = len(models)
    
    if n_models < 2:
        print("Need at least 2 models for comparison visualizations")
        return {}
    
    print("Creating advanced model comparison visualizations...")
    
    # 1. Comprehensive Performance Radar Chart
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Radar chart data preparation
    metrics_radar = ['Top-1 Acc', 'Top-5 Acc', 'F1 Macro', 'Precision', 'Recall', 'Balanced Acc']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics_radar), endpoint=False)
    angles = np.concatenate((angles, [angles[0]]))  # Complete the circle
    
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, (model, color) in enumerate(zip(models, colors)):
        result = all_results[model]
        metrics = result['metrics']
        
        # Collect metrics (normalize to 0-1 scale)
        values = [
            metrics.get('top1_accuracy', 0),
            metrics.get('top5_accuracy', 0), 
            metrics.get('f1_macro', 0),
            metrics.get('precision_macro', 0),
            metrics.get('recall_macro', 0),
            metrics.get('balanced_accuracy', 0)
        ]
        values += [values[0]]  # Complete the circle
        
        ax1.plot(angles, values, 'o-', linewidth=2, label=model.upper(), color=color)
        ax1.fill(angles, values, alpha=0.25, color=color)
    
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(metrics_radar)
    ax1.set_ylim(0, 1)
    ax1.set_title('Model Performance Radar Chart', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
    ax1.grid(True)
    
    # 2. Metric Correlation Heatmap
    metric_names = ['top1_acc', 'top5_acc', 'f1_macro', 'precision_macro', 'recall_macro', 'balanced_acc']
    metric_matrix = []
    
    for model in models:
        metrics = all_results[model]['metrics']
        model_metrics = [
            metrics.get('top1_accuracy', 0),
            metrics.get('top5_accuracy', 0),
            metrics.get('f1_macro', 0),
            metrics.get('precision_macro', 0), 
            metrics.get('recall_macro', 0),
            metrics.get('balanced_accuracy', 0)
        ]
        metric_matrix.append(model_metrics)
    
    metric_matrix = np.array(metric_matrix)
    correlation_matrix = np.corrcoef(metric_matrix.T)
    
    im = ax2.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    ax2.set_xticks(range(len(metric_names)))
    ax2.set_yticks(range(len(metric_names)))
    ax2.set_xticklabels([name.replace('_', ' ').title() for name in metric_names], rotation=45)
    ax2.set_yticklabels([name.replace('_', ' ').title() for name in metric_names])
    ax2.set_title('Metric Correlation Matrix', fontsize=14, fontweight='bold')
    
    # Add correlation values
    for i in range(len(metric_names)):
        for j in range(len(metric_names)):
            text = ax2.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2, shrink=0.6)
    
    # 3. Per-Class Performance Variance
    categories = all_results[models[0]]['dataset_info']['all_categories']
    class_variances = []
    class_means = []
    
    for category in categories:
        class_accs = []
        for model in models:
            per_class = all_results[model]['metrics']['per_class_accuracy']
            acc = per_class.get(category, 0)
            class_accs.append(acc * 100)
        
        if len(class_accs) > 1:
            class_variances.append(np.var(class_accs))
            class_means.append(np.mean(class_accs))
        else:
            class_variances.append(0)
            class_means.append(class_accs[0] if class_accs else 0)
    
    # Sort by variance for better visualization
    sorted_indices = np.argsort(class_variances)[::-1]
    top_variance_classes = [categories[i] for i in sorted_indices[:10]]
    top_variances = [class_variances[i] for i in sorted_indices[:10]]
    
    bars = ax3.bar(range(len(top_variance_classes)), top_variances, color='orange', alpha=0.7)
    ax3.set_xlabel('Action Classes')
    ax3.set_ylabel('Performance Variance')
    ax3.set_title('Classes with Highest Model Disagreement', fontsize=14, fontweight='bold')
    ax3.set_xticks(range(len(top_variance_classes)))
    ax3.set_xticklabels(top_variance_classes, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, var in zip(bars, top_variances):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{var:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Model Agreement Matrix
    if n_models >= 2:
        agreement_matrix = np.zeros((n_models, n_models))
        
        # This would require individual model predictions - simplified version
        for i in range(n_models):
            for j in range(n_models):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    # Estimate agreement from accuracy similarity
                    acc_i = all_results[models[i]]['metrics']['top1_accuracy']
                    acc_j = all_results[models[j]]['metrics']['top1_accuracy']
                    agreement_matrix[i, j] = 1 - abs(acc_i - acc_j)  # Simple similarity measure
        
        im4 = ax4.imshow(agreement_matrix, cmap='viridis', vmin=0, vmax=1)
        ax4.set_xticks(range(n_models))
        ax4.set_yticks(range(n_models))
        ax4.set_xticklabels([m.upper() for m in models])
        ax4.set_yticklabels([m.upper() for m in models])
        ax4.set_title('Model Agreement (Similarity)', fontsize=14, fontweight='bold')
        
        # Add agreement values
        for i in range(n_models):
            for j in range(n_models):
                text = ax4.text(j, i, f'{agreement_matrix[i, j]:.2f}',
                               ha="center", va="center", color="white", fontweight='bold')
        
        plt.colorbar(im4, ax=ax4, shrink=0.6)
    
    plt.tight_layout()
    comparison_viz_path = os.path.join(comparison_dir, f'advanced_model_comparison_{timestamp}.png')
    plt.savefig(comparison_viz_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return {
        'advanced_comparison': comparison_viz_path
    }

if __name__ == "__main__":
        parser = argparse.ArgumentParser(description="Evaluate a trained action recognition model.")
        parser.add_argument("model_type", type=str, nargs="?", default="timesformer", help="Model type to evaluate (e.g., timesformer, vit, videomae)")
        parser.add_argument("use_processed", type=str, nargs="?", default="True", help="Whether to use processed data (True/False)")
        parser.add_argument("--model-path", type=str, default=None, help="Path to a specific model checkpoint (.pth) file")
        args = parser.parse_args()

        if args.model_type.lower() == 'all':
            use_processed = args.use_processed.lower() == 'true'
            print(f"Starting evaluation for ALL MODELS")
            print(f"Use processed data: {use_processed}")
            print("="*70)
            try:
                comparison_results = evaluate_all_models(use_processed)
                print("="*70)
                print(f"All model evaluations completed successfully!")
            except Exception as e:
                print(f"ERROR: Evaluation failed with error: {str(e)}")
                raise
        else:
            model_type = args.model_type.lower()
            use_processed = args.use_processed.lower() == 'true'
            model_path = args.model_path
            print(f"Starting evaluation for model: {model_type.upper()}")
            print(f"Use processed data: {use_processed}")
            if model_path:
                print(f"Using custom model path: {model_path}")
            print("="*60)
            try:
                results = evaluate_model(model_type, use_processed, model_path=model_path)
                if results:
                    print("="*60)
                    print(f"Evaluation completed successfully!")
                    print(f"Final Accuracy: {results['metrics']['top1_accuracy']*100:.2f}%")
                else:
                    print("ERROR: Evaluation failed - model not found or not trained")
            except Exception as e:
                print(f"ERROR: Evaluation failed with error: {str(e)}")
                raise

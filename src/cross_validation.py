"""
Cross-validation training for small datasets to get more reliable evaluation
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from pathlib import Path
import json
import os
import random

# Simple KFold implementation - no external dependencies needed
class KFold:
    def __init__(self, n_splits=5, shuffle=False, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X):
        n_samples = len(X)
        indices = list(X)
        
        if self.shuffle:
            if self.random_state is not None:
                random.seed(self.random_state)
            random.shuffle(indices)
        
        fold_sizes = [n_samples // self.n_splits] * self.n_splits
        for i in range(n_samples % self.n_splits):
            fold_sizes[i] += 1
        
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_indices = indices[start:stop]
            train_indices = indices[:start] + indices[stop:]
            yield train_indices, test_indices
            current = stop

from dataset import HMDBDataset
from train import EarlyStopping

# Simple config loading without yaml dependency
def load_config_simple():
    """Load basic config without yaml - use hardcoded values"""
    return {
        'training': {
            'batch_size': 4,
            'learning_rate': 0.0001,
            'epochs': 20,
            'weight_decay': 0.01,
            'patience': 3
        },
        'dataset': {
            'num_frames': 8,
            'frame_size': 224,
            'sampling_rate': 32
        }
    }

def cross_validate_model(model_class, model_config, config, k_folds=5):
    """
    Perform k-fold cross-validation on a small dataset
    """
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION: {model_config['model_name']} ({k_folds} folds)")
    print(f"{'='*50}")
    
    # Load full dataset (no splits - use mode without train/val/test filtering)
    full_dataset = HMDBDataset(
        root_dir=None,  # Will use config default
        categories=None,  # Will use config default
        num_frames=config.get('data', {}).get('num_frames', 8),
        frame_size=config.get('data', {}).get('frame_size', 224),
        sampling_rate=config.get('data', {}).get('sampling_rate', 32),
        mode='all',  # Custom mode to get all data
        use_processed=True
    )
    
    print(f"Total samples: {len(full_dataset)}")
    print(f"Classes: {full_dataset.categories}")
    
    if len(full_dataset) < k_folds:
        print(f"WARNING: Dataset too small ({len(full_dataset)}) for {k_folds}-fold CV")
        k_folds = max(2, len(full_dataset) // 2)
        print(f"Reducing to {k_folds}-fold CV")
    
    # Setup k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
        print(f"\n--- FOLD {fold + 1}/{k_folds} ---")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create data loaders for this fold
        train_subset = Subset(full_dataset, train_idx)
        val_subset = Subset(full_dataset, val_idx)
        
        train_loader = DataLoader(
            train_subset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_subset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        # Initialize model
        model = model_class(model_config)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training'].get('weight_decay', 0.01)
        )
        
        early_stopping = EarlyStopping(
            patience=config['training'].get('patience', 3),  # Shorter patience for CV
            verbose=False
        )
        
        # Training loop for this fold
        fold_train_losses = []
        fold_val_losses = []
        fold_val_accuracies = []
        
        for epoch in range(config['training']['epochs']):
            # Training
            model.train()
            train_loss = 0.0
            for batch_idx, batch in enumerate(train_loader):
                # Handle different data formats
                if isinstance(batch, dict):
                    data = batch['pixel_values']
                    targets = batch['labels']
                else:
                    data, targets = batch
                
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(train_loader)
            fold_train_losses.append(avg_train_loss)
            
            # Validation
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    # Handle different data formats
                    if isinstance(batch, dict):
                        data = batch['pixel_values']
                        targets = batch['labels']
                    else:
                        data, targets = batch
                    
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = 100 * correct / total
            
            fold_val_losses.append(avg_val_loss)
            fold_val_accuracies.append(val_accuracy)
            
            if epoch % 5 == 0:
                print(f"Epoch {epoch:3d}: Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
            
            # Early stopping check
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Record fold results
        fold_result = {
            'fold': fold + 1,
            'final_val_accuracy': fold_val_accuracies[-1],
            'best_val_accuracy': max(fold_val_accuracies),
            'final_val_loss': fold_val_losses[-1],
            'epochs_trained': len(fold_val_losses),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        fold_results.append(fold_result)
        
        print(f"Fold {fold + 1} Results:")
        print(f"  Final Val Accuracy: {fold_result['final_val_accuracy']:.2f}%")
        print(f"  Best Val Accuracy: {fold_result['best_val_accuracy']:.2f}%")
        print(f"  Epochs Trained: {fold_result['epochs_trained']}")
    
    # Calculate cross-validation statistics
    final_accuracies = [r['final_val_accuracy'] for r in fold_results]
    best_accuracies = [r['best_val_accuracy'] for r in fold_results]
    
    cv_results = {
        'model_name': model_config['model_name'],
        'k_folds': k_folds,
        'fold_results': fold_results,
        'final_accuracy_mean': np.mean(final_accuracies),
        'final_accuracy_std': np.std(final_accuracies),
        'best_accuracy_mean': np.mean(best_accuracies),
        'best_accuracy_std': np.std(best_accuracies),
        'config': config
    }
    
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION SUMMARY")
    print(f"{'='*50}")
    print(f"Final Accuracy: {cv_results['final_accuracy_mean']:.2f}% ± {cv_results['final_accuracy_std']:.2f}%")
    print(f"Best Accuracy:  {cv_results['best_accuracy_mean']:.2f}% ± {cv_results['best_accuracy_std']:.2f}%")
    
    # Save results
    results_dir = Path('results/cross_validation')
    results_dir.mkdir(parents=True, exist_ok=True)
    
    results_file = results_dir / f"cv_{model_config['model_name']}_k{k_folds}.json"
    with open(results_file, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    print(f"Results saved to: {results_file}")
    
    return cv_results

def main():
    """Run cross-validation experiments"""
    config = load_config_simple()  # Use simple config without yaml
    
    # Check dataset size first - use HMDBDataset directly
    temp_dataset = HMDBDataset(mode='all', use_processed=True)
    dataset_size = len(temp_dataset)
    
    print(f"Dataset size: {dataset_size} samples")
    print(f"Categories: {temp_dataset.categories}")
    
    if dataset_size < 20:
        print("Dataset very small - using 3-fold CV")
        k_folds = 3
    elif dataset_size < 50:
        print("Dataset small - using 5-fold CV")
        k_folds = 5
    else:
        print("Dataset adequate - using standard train/val/test split")
        print("Cross-validation not necessary")
        return
    
    # Simple model testing - just test one model to start
    print(f"\nTesting cross-validation with {k_folds} folds...")
    print("Testing basic model structure...")
    
    # Create a simple test model for cross-validation
    class SimpleTestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.flatten = nn.Flatten()
            self.fc = nn.Linear(8 * 3 * 224 * 224, len(temp_dataset.categories))
        
        def forward(self, x):
            if isinstance(x, dict) and 'pixel_values' in x:
                x = x['pixel_values']
            x = self.flatten(x)
            return self.fc(x)
    
    # Test cross-validation with simple model
    model_config = {'model_name': 'simple_test'}
    
    try:
        cv_results = cross_validate_simple_model(model_config, config, temp_dataset, k_folds)
        
        print(f"\n{'='*50}")
        print(f"CROSS-VALIDATION TEST COMPLETE")
        print(f"{'='*50}")
        print(f"Final Accuracy: {cv_results['final_accuracy_mean']:.2f}% ± {cv_results['final_accuracy_std']:.2f}%")
        print(f"Best Accuracy:  {cv_results['best_accuracy_mean']:.2f}% ± {cv_results['best_accuracy_std']:.2f}%")
        
        # Interpret results
        if cv_results['final_accuracy_std'] > 10:
            print("WARNING: High variance detected - likely overfitting")
        elif cv_results['final_accuracy_mean'] > 95:
            print("WARNING: Very high accuracy - possible memorization")
        else:
            print("Reasonable results - cross-validation working")
            
    except Exception as e:
        print(f"ERROR: Cross-validation failed: {e}")
        import traceback
        traceback.print_exc()

def cross_validate_simple_model(model_config, config, dataset, k_folds=5):
    """Simplified cross-validation for testing"""
    print(f"\n{'='*50}")
    print(f"CROSS-VALIDATION: {model_config['model_name']} ({k_folds} folds)")
    print(f"{'='*50}")
    
    # Setup k-fold cross-validation
    kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(dataset)))):
        print(f"\n--- FOLD {fold + 1}/{k_folds} ---")
        print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
        
        # Create data loaders for this fold
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        train_loader = DataLoader(train_subset, batch_size=2, shuffle=True)  # Small batch for testing
        val_loader = DataLoader(val_subset, batch_size=2, shuffle=False)
        
        # Simple model for testing
        from torch import nn
        class SimpleModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.flatten = nn.Flatten()
                self.fc = nn.Linear(8 * 3 * 224 * 224, num_classes)
                
            def forward(self, x):
                if isinstance(x, dict) and 'pixel_values' in x:
                    x = x['pixel_values']
                x = self.flatten(x)
                return self.fc(x)
        
        model = SimpleModel(len(dataset.categories))
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        
        # Setup training
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
        
        # Short training for testing (only 5 epochs)
        fold_val_accuracies = []
        
        for epoch in range(5):  # Short test
            # Training
            model.train()
            for batch in train_loader:
                if isinstance(batch, dict):
                    data = batch['pixel_values']
                    targets = batch['labels']
                else:
                    data, targets = batch
                
                data, targets = data.to(device), targets.to(device)
                
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Validation
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    if isinstance(batch, dict):
                        data = batch['pixel_values']
                        targets = batch['labels']
                    else:
                        data, targets = batch
                    
                    data, targets = data.to(device), targets.to(device)
                    outputs = model(data)
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total += targets.size(0)
                    correct += (predicted == targets).sum().item()
            
            val_accuracy = 100 * correct / total
            fold_val_accuracies.append(val_accuracy)
            
            if epoch == 4:  # Last epoch
                print(f"Final Val Accuracy: {val_accuracy:.2f}%")
        
        # Record fold results
        fold_result = {
            'fold': fold + 1,
            'final_val_accuracy': fold_val_accuracies[-1],
            'best_val_accuracy': max(fold_val_accuracies),
            'epochs_trained': len(fold_val_accuracies),
            'train_samples': len(train_idx),
            'val_samples': len(val_idx)
        }
        fold_results.append(fold_result)
    
    # Calculate statistics
    final_accuracies = [r['final_val_accuracy'] for r in fold_results]
    best_accuracies = [r['best_val_accuracy'] for r in fold_results]
    
    cv_results = {
        'model_name': model_config['model_name'],
        'k_folds': k_folds,
        'fold_results': fold_results,
        'final_accuracy_mean': np.mean(final_accuracies),
        'final_accuracy_std': np.std(final_accuracies),
        'best_accuracy_mean': np.mean(best_accuracies),
        'best_accuracy_std': np.std(best_accuracies)
    }
    
    return cv_results

if __name__ == "__main__":
    main()

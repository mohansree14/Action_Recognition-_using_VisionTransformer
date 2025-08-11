# train.py
# Model training script

import torch
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor
import sys
import os
import time
from datetime import datetime
import json
import numpy as np
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.vit_model import get_timesformer_model, get_vit_model, get_videomae_model
from dataset import HMDBDataset
import yaml

class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=7, min_delta=0, verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = None
        self.counter = 0
        self.early_stop = False
        
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True

def calculate_accuracy(outputs, labels):
    """Calculate accuracy from model outputs and labels"""
    predictions = torch.argmax(outputs.logits, dim=1)
    correct = (predictions == labels).float()
    accuracy = correct.mean().item()
    return accuracy

def validate_model(model, val_loader, device, extractor, model_type="timesformer"):
    """Validate the model on validation set"""
    model.eval()
    total_val_loss = 0
    total_val_accuracy = 0
    num_val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            pixel_values = batch['pixel_values']
            labels = batch['labels']
            
            # Move data to GPU
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Handle different input preprocessing for different models
            if model_type == "vit":
                # ViT: Pass video data directly to model
                inputs = {"pixel_values": pixel_values}
            else:
                # TimeSFormer/VideoMAE: Convert to numpy for video feature extractor
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
            outputs = model(**inputs, labels=labels)
            
            # Calculate loss and accuracy
            val_loss = outputs.loss.item()
            val_accuracy = calculate_accuracy(outputs, labels)
            
            total_val_loss += val_loss
            total_val_accuracy += val_accuracy
            num_val_batches += 1
    
    model.train()  # Set back to training mode
    avg_val_loss = total_val_loss / num_val_batches
    avg_val_accuracy = total_val_accuracy / num_val_batches
    
    return avg_val_loss, avg_val_accuracy

def train_model(model_type="timesformer", use_processed=True, config_file=None, 
               learning_rate=None, num_frames=None, sampling_rate=None, batch_size=None, epochs=None,
               experiment_name=None, optimizer_type=None, patience=None):
    # Set device - use GPU if available, otherwise CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Detailed GPU diagnostics and verification
    if torch.cuda.is_available():
        print(f"   CUDA is available!")
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   CUDA Version: {torch.version.cuda}")
        print(f"   PyTorch CUDA Version: {torch.version.cuda}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"   Current GPU: {torch.cuda.current_device()}")
        print(f"   GPU Count: {torch.cuda.device_count()}")
        
        # Test GPU functionality
        test_tensor = torch.randn(10, 10).to(device)
        print(f"   GPU Test: {test_tensor.device} - {'SUCCESS' if test_tensor.is_cuda else 'FAILED'}")
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        print(f"   GPU cache cleared")
        
        # Force GPU usage verification
        if not test_tensor.is_cuda:
            print("   WARNING: GPU test failed, falling back to CPU")
            device = torch.device('cpu')
        else:
            print(f"   GPU acceleration confirmed - training will be fast!")
    else:
        print("   CUDA not available - training will be slow!")
        print("   Make sure GPU accelerator is enabled in Kaggle settings")
        print("   Or check CUDA installation if running locally")

    # Load configuration with support for different config files
    if config_file is None:
        config_file = "configs/coursework_config.yaml"
    
    print(f"Loading configuration from: {config_file}")
    with open(config_file) as f:
        config = yaml.safe_load(f)
    
    # Apply experiment configuration if specified
    if experiment_name is not None:
        if 'experiments' in config and experiment_name in config['experiments']:
            experiment_config = config['experiments'][experiment_name]
            print(f"Loading experiment: '{experiment_name}'")
            print(f"   Description: {experiment_config.get('description', 'No description')}")
            
            # Override config with experiment settings
            if 'learning_rate' in experiment_config:
                config['training']['learning_rate'] = experiment_config['learning_rate']
            if 'num_frames' in experiment_config:
                config['dataset']['num_frames'] = experiment_config['num_frames']
            if 'sampling_rate' in experiment_config:
                config['dataset']['sampling_rate'] = experiment_config['sampling_rate']
            if 'batch_size' in experiment_config:
                config['training']['batch_size'] = experiment_config['batch_size']
            if 'epochs' in experiment_config:
                config['training']['epochs'] = experiment_config['epochs']
            
            print(f"   Applied experiment settings:")
            print(f"      Learning Rate: {experiment_config.get('learning_rate', 'unchanged')}")
            print(f"      Frames: {experiment_config.get('num_frames', 'unchanged')}")
            print(f"      Sampling Rate: {experiment_config.get('sampling_rate', 'unchanged')}")
            print(f"      Batch Size: {experiment_config.get('batch_size', 'unchanged')}")
            print(f"      Epochs: {experiment_config.get('epochs', 'unchanged')}")
        else:
            print(f"Warning: Experiment '{experiment_name}' not found in config")
            if 'experiments' in config:
                available = list(config['experiments'].keys())
                print(f"   Available experiments: {', '.join(available)}")
            else:
                print("   No experiments section found in config")
    
    # Override config parameters if provided (command line overrides take priority)
    if learning_rate is not None:
        config['training']['learning_rate'] = learning_rate
        print(f"Command line override - learning rate: {learning_rate}")
    
    if num_frames is not None:
        config['dataset']['num_frames'] = num_frames
        print(f"Command line override - num_frames: {num_frames}")
    
    if sampling_rate is not None:
        config['dataset']['sampling_rate'] = sampling_rate
        print(f"Command line override - sampling_rate: {sampling_rate}")
    
    if batch_size is not None:
        config['training']['batch_size'] = batch_size
        print(f"Command line override - batch_size: {batch_size}")
    
    if epochs is not None:
        config['training']['epochs'] = epochs
        print(f"Command line override - epochs: {epochs}")
    
    if optimizer_type is not None:
        config['training']['optimizer_type'] = optimizer_type
        print(f"Command line override - optimizer: {optimizer_type}")
    
    categories = config['dataset']['categories']
    
    # Determine which dataset path to use
    original_data_path = config['dataset']['root_dir']
    processed_data_path = os.path.join("results", "HMDB_simp_processed")
    
    if use_processed and os.path.exists(processed_data_path):
        dataset_path = processed_data_path
        print(f"USING PROCESSED DATASET: {os.path.abspath(dataset_path)}")
        print(f"   Preprocessed data with augmentations")
    else:
        dataset_path = original_data_path
        print(f"USING ORIGINAL DATASET: {dataset_path}")
        print(f"   Raw data without preprocessing")
    
    print(f"   Expected categories: {len(categories)}")
    print(f"   Dataset path exists: {os.path.exists(dataset_path)}")
    
    # Get frame parameters from config (with defaults if not specified)
    num_frames = config['dataset'].get('num_frames', 8)
    frame_size = config['dataset'].get('frame_size', 224)
    sampling_rate = config['dataset'].get('sampling_rate', 32)
    
    print(f"   Frame parameters:")
    print(f"      Number of frames: {num_frames}")
    print(f"      Frame size: {frame_size}x{frame_size}")
    print(f"      Sampling rate: {sampling_rate}")
    
    # Create datasets with frame parameters
    train_dataset = HMDBDataset(root_dir=dataset_path, categories=categories, 
                               num_frames=num_frames, frame_size=frame_size, 
                               sampling_rate=sampling_rate, mode='train', use_processed=False)
    val_dataset = HMDBDataset(root_dir=dataset_path, categories=categories,
                             num_frames=num_frames, frame_size=frame_size,
                             sampling_rate=sampling_rate, mode='val', use_processed=False)
    
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, pin_memory=True)

    # Load feature extractor based on model type
    if model_type == "vit":
        # ViT uses image feature extractor
        extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        print(f"Loaded ViT image feature extractor")
    else:
        # TimeSFormer and VideoMAE use video feature extractor
        extractor = AutoFeatureExtractor.from_pretrained(config['model']['model_name'])
        print(f"Loaded video feature extractor: {config['model']['model_name']}")
    if model_type == "timesformer":
        model = get_timesformer_model(num_classes=len(categories))
    elif model_type == "vit":
        model = get_vit_model(num_classes=len(categories))
    elif model_type == "videomae":
        model = get_videomae_model(num_classes=len(categories))
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Move model to GPU
    model = model.to(device)
    print(f"   Model moved to {device}")
    print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Verify model is on GPU
    if torch.cuda.is_available() and device.type == 'cuda':
        model_device = next(model.parameters()).device
        print(f"   Model device verification: {model_device}")
        if model_device.type != 'cuda':
            print("   WARNING: Model is not on GPU!")
        else:
            print(f"   Model successfully on GPU: {model_device}")
    
    model.train()
    
    # Early stopping configuration - increased patience for longer training
    patience = 20  # Stop if no improvement for 20 epochs (or set to epochs//2 for very long training)
    
    # Get optimizer configuration from config
    optimizer_config = config.get('optimizer', {})
    optimizer_name = config['training'].get('optimizer_type', optimizer_config.get('type', 'AdamW'))
    weight_decay = config['training'].get('weight_decay', 0.01)
    learning_rate_val = config['training']['learning_rate']
    
    # Create optimizer based on type
    if optimizer_name.lower() == 'adamw':
        # Ensure betas and eps are proper numeric types
        betas = optimizer_config.get('betas', [0.9, 0.999])
        if isinstance(betas, list):
            betas = [float(b) for b in betas]
        eps = float(optimizer_config.get('eps', 1e-8))
        
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate_val,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif optimizer_name.lower() == 'adam':
        # Ensure betas and eps are proper numeric types
        betas = optimizer_config.get('betas', [0.9, 0.999])
        if isinstance(betas, list):
            betas = [float(b) for b in betas]
        eps = float(optimizer_config.get('eps', 1e-8))
        
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=learning_rate_val,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps
        )
    elif optimizer_name.lower() == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=learning_rate_val,
            weight_decay=weight_decay,
            momentum=0.9,
            nesterov=True
        )
    elif optimizer_name.lower() == 'rmsprop':
        # Ensure alpha and eps are proper numeric types
        alpha = float(optimizer_config.get('alpha', 0.99))
        eps = float(optimizer_config.get('eps', 1e-8))
        
        optimizer = torch.optim.RMSprop(
            model.parameters(), 
            lr=learning_rate_val,
            weight_decay=weight_decay,
            alpha=alpha,
            eps=eps
        )
    else:
        # Default to AdamW
        print(f"Unknown optimizer '{optimizer_name}', defaulting to AdamW")
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=learning_rate_val,
            weight_decay=weight_decay
        )
        optimizer_name = 'AdamW'
    
    print(f"   Optimizer: {optimizer_name} with weight_decay={weight_decay}, lr={learning_rate_val}")
    
    # Configure early stopping patience
    if patience is None:
        # Default patience based on epochs
        total_epochs = config['training']['epochs']
        patience = max(10, total_epochs // 5)  # At least 10, or 20% of total epochs
    elif patience == 0:
        patience = float('inf')  # Disable early stopping completely
        print(f"   Early stopping: DISABLED")
    else:
        print(f"   Early stopping patience: {patience} epochs")

    # Create experiment-specific filename with parameters
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create descriptive filename based on experiment parameters (with fallbacks)
    lr = config['training'].get('learning_rate', 0.0001)
    frames = config['dataset'].get('num_frames', 8)
    sampling = config['dataset'].get('sampling_rate', 32)
    
    lr_str = f"lr{lr:.5f}".replace('.', '')
    frames_str = f"f{frames}"
    sampling_str = f"s{sampling}"
    
    experiment_id = f"{lr_str}_{frames_str}_{sampling_str}"
    
    model_dir = os.path.join("results", f"{model_type}_model")
    os.makedirs(model_dir, exist_ok=True)
    
    log_path = os.path.join(model_dir, f"{model_type}_{experiment_id}_{timestamp}.log")
    model_path = os.path.join(model_dir, f"{model_type}_{experiment_id}_{timestamp}.pth")
    config_path = os.path.join(model_dir, f"{model_type}_{experiment_id}_config_{timestamp}.json")
    metrics_path = os.path.join(model_dir, f"{model_type}_{experiment_id}_metrics_{timestamp}.json")
    
    # Save training configuration with all parameters
    training_config = {
        "model_type": model_type,
        "use_processed": use_processed,
        "dataset_path": dataset_path,
        "num_categories": len(categories),
        "categories": categories,
        "training_params": config['training'],
        "dataset_params": config['dataset'],  # Include all dataset parameters
        "model_params": config['model'],
        "total_parameters": sum(p.numel() for p in model.parameters()),
        "timestamp": timestamp,
        "device": str(device),
        "experiment_name": experiment_name,
        "parameter_overrides": {
            "learning_rate": learning_rate,
            "num_frames": num_frames,
            "sampling_rate": sampling_rate,
            "batch_size": batch_size,
            "epochs": epochs,
            "optimizer_type": optimizer_type
        },
        "experiment_settings": {
            "learning_rate": config['training'].get('learning_rate', 0.0001),
            "num_frames": config['dataset'].get('num_frames', 8),
            "frame_size": config['dataset'].get('frame_size', 224),
            "sampling_rate": config['dataset'].get('sampling_rate', 32),
            "batch_size": config['training'].get('batch_size', 4),
            "epochs": config['training'].get('epochs', 10)
        }
    }
    
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"   Model directory: {model_dir}")
    print(f"   Training config saved: {config_path}")
    
    # Print experiment details
    print(f"\nEXPERIMENT DETAILS:")
    if experiment_name:
        print(f"   Experiment Name: {experiment_name}")
    print(f"   Learning Rate: {config['training'].get('learning_rate', 0.0001)}")
    print(f"   Frames: {config['dataset'].get('num_frames', 8)}")
    print(f"   Sampling Rate: {config['dataset'].get('sampling_rate', 32)}")
    print(f"   Batch Size: {config['training'].get('batch_size', 4)}")
    print(f"   Epochs: {config['training'].get('epochs', 10)}")
    print(f"   Frame Size: {config['dataset'].get('frame_size', 224)}x{config['dataset'].get('frame_size', 224)}")
    print(f"   Experiment ID: {experiment_id}")

    # Training tracking variables with early stopping
    best_val_accuracy = 0.0
    best_epoch = 0
    patience_counter = 0
    early_stop = False
    
    training_metrics = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracies": [],
        "gpu_memory_usage": [],
        "epochs_completed": 0,
        "experiment_id": experiment_id,
        "experiment_name": experiment_name,
        "early_stopped": False,
        "early_stop_epoch": None,
        "experiment_details": {
            "learning_rate": config['training'].get('learning_rate', 0.0001),
            "num_frames": config['dataset'].get('num_frames', 8),
            "sampling_rate": config['dataset'].get('sampling_rate', 32),
            "batch_size": config['training'].get('batch_size', 4),
            "epochs": config['training'].get('epochs', 10),
            "frame_size": config['dataset'].get('frame_size', 224),
            "patience": patience
        }
    }
    
    for epoch in range(config['training']['epochs']):
        epoch_start_time = time.time()
        total_loss = 0
        num_batches = 0
        
        print(f"\nStarting Epoch {epoch+1}/{config['training']['epochs']}")
        
        for batch_idx, batch in enumerate(train_loader):
            pixel_values = batch['pixel_values']  # Shape: [batch_size, num_frames, channels, height, width]
            labels = batch['labels']
            
            # Move data to GPU with non_blocking for faster transfer
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            # Handle different input preprocessing for different models
            if model_type == "vit":
                # ViT: Pass video data directly to model, it will handle frame selection
                inputs = {"pixel_values": pixel_values}
            else:
                # TimeSFormer/VideoMAE: Convert to numpy for video feature extractor
                # Feature extractor expects: [batch_size, num_frames, height, width, channels]
                videos_numpy = []
                pixel_values_cpu = pixel_values.cpu()  # Move to CPU once for numpy conversion
                for video in pixel_values_cpu:
                    # video shape: [num_frames, channels, height, width]
                    # Convert to [num_frames, height, width, channels] and denormalize
                    video_frames = []
                    for frame in video:
                        # frame shape: [channels, height, width]
                        frame_np = frame.permute(1, 2, 0).numpy()  # [height, width, channels]
                        frame_np = (frame_np * 255).astype('uint8')  # denormalize to 0-255
                        video_frames.append(frame_np)
                    videos_numpy.append(video_frames)
                
                # Process with feature extractor
                inputs = extractor(videos_numpy, return_tensors="pt")
            
            # Move inputs to GPU with better error handling
            inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # Ensure labels are on correct device
            labels = labels.to(device, non_blocking=True)
            
            # Forward pass
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            
            # Calculate training accuracy for this batch
            train_accuracy = calculate_accuracy(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            num_batches += 1
            
            # Print progress every 10 batches
            if batch_idx % 10 == 0:
                if torch.cuda.is_available() and device.type == 'cuda':
                    gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
                    gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3
                    
                    # Verify tensors are on GPU
                    data_on_gpu = pixel_values.is_cuda and labels.is_cuda
                    model_on_gpu = next(model.parameters()).is_cuda
                    
                    gpu_status = "GPU" if (data_on_gpu and model_on_gpu) else "CPU"
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}, Acc = {train_accuracy:.3f}, {gpu_status}: {gpu_memory:.1f}GB/{gpu_reserved:.1f}GB")
                    
                    if not (data_on_gpu and model_on_gpu):
                        print(f"   WARNING: Data on GPU: {data_on_gpu}, Model on GPU: {model_on_gpu}")
                else:
                    print(f"Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}, Acc = {train_accuracy:.3f}, Device: CPU")
        
        # Calculate epoch metrics
        epoch_loss = total_loss / num_batches
        epoch_time = time.time() - epoch_start_time
        
        # Validate model
        print(f"   Running validation...")
        val_loss, val_accuracy = validate_model(model, val_loader, device, extractor, model_type)
        
        # Track metrics
        training_metrics["train_losses"].append(epoch_loss)
        training_metrics["val_losses"].append(val_loss)
        training_metrics["val_accuracies"].append(val_accuracy)
        training_metrics["epochs_completed"] = epoch + 1
        
        # Check if this is the best model so far
        is_best = val_accuracy > best_val_accuracy
        if is_best:
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter
            # Save best model
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_accuracy': val_accuracy,
                'val_loss': val_loss,
                'train_loss': epoch_loss,
                'model_type': model_type,
                'config': training_config
            }, model_path)
            print(f"   New best model saved! Accuracy: {val_accuracy:.3f}")
        else:
            patience_counter += 1
            print(f"   No improvement for {patience_counter}/{patience} epochs")
            
            # Check for early stopping
            if patience_counter >= patience:
                print(f"   Early stopping triggered! No improvement for {patience} epochs")
                early_stop = True
                training_metrics["early_stopped"] = True
                training_metrics["early_stop_epoch"] = epoch + 1
        
        # Log results
        if torch.cuda.is_available() and device.type == 'cuda':
            gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
            gpu_reserved = torch.cuda.memory_reserved(device) / 1024**3
            training_metrics["gpu_memory_usage"].append(gpu_memory)
            
            # Verify GPU usage
            model_on_gpu = next(model.parameters()).is_cuda
            gpu_status = "GPU" if model_on_gpu else "CPU"
            
            log_message = (f"Epoch {epoch+1}/{config['training']['epochs']}: "
                          f"Train Loss = {epoch_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, "
                          f"Val Accuracy = {val_accuracy:.3f}, "
                          f"Time = {epoch_time:.1f}s, "
                          f"{gpu_status}: {gpu_memory:.1f}GB/{gpu_reserved:.1f}GB"
                          f"{' (BEST!)' if is_best else ''}")
            
            print(f"{log_message}")
            
            with open(log_path, "a") as logf:
                logf.write(f"{log_message}\n")
            
            # Clear GPU cache after each epoch
            torch.cuda.empty_cache()
        else:
            log_message = (f"Epoch {epoch+1}/{config['training']['epochs']}: "
                          f"Train Loss = {epoch_loss:.4f}, "
                          f"Val Loss = {val_loss:.4f}, "
                          f"Val Accuracy = {val_accuracy:.3f}, "
                          f"Time = {epoch_time:.1f}s, "
                          f"Device: CPU"
                          f"{' (BEST!)' if is_best else ''}")
            
            print(f"{log_message}")
            
            with open(log_path, "a") as logf:
                logf.write(f"{log_message}\n")
        
        # Save metrics after each epoch
        with open(metrics_path, 'w') as f:
            json.dump(training_metrics, f, indent=2)
        
        # Check for early stopping
        if early_stop:
            print(f"\nTraining stopped early at epoch {epoch + 1}")
            print(f"   Best model was at epoch {best_epoch} with accuracy {best_val_accuracy:.3f}")
            break
    
    # Training completed - save final metrics and summary
    final_summary = {
        "training_completed": True,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_val_accuracy,
        "final_train_loss": training_metrics["train_losses"][-1],
        "final_val_loss": training_metrics["val_losses"][-1],
        "final_val_accuracy": training_metrics["val_accuracies"][-1],
        "total_epochs": config['training']['epochs'],
        "model_saved_path": model_path,
        "training_time_per_epoch": epoch_time,
        "model_type": model_type
    }
    
    # Add final summary to metrics
    training_metrics["summary"] = final_summary
    
    with open(metrics_path, 'w') as f:
        json.dump(training_metrics, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"   Best validation accuracy: {best_val_accuracy:.3f} (Epoch {best_epoch})")
    print(f"   Best model saved: {model_path}")
    print(f"   Training metrics: {metrics_path}")
    print(f"   Training log: {log_path}")
    
    return model_path, best_val_accuracy, training_metrics

if __name__ == "__main__":
    import sys
    import argparse
    
    # Check if we're using old-style positional arguments or new-style arguments
    using_old_style = (len(sys.argv) > 1 and 
                      not sys.argv[1].startswith('-') and 
                      sys.argv[1] in ['timesformer', 'vit', 'videomae'])
    
    if using_old_style:
        # Old style: python train.py timesformer true [config_file]
        model_type = sys.argv[1].lower() if len(sys.argv) > 1 else 'timesformer'
        use_processed = sys.argv[2].lower() == 'true' if len(sys.argv) > 2 else True
        config_file = sys.argv[3] if len(sys.argv) > 3 else None
        
        # No parameter overrides in old style
        learning_rate = None
        num_frames = None
        sampling_rate = None
        batch_size = None
        epochs = None
        experiment_name = None
        optimizer_type = None
        patience = None
        
        print(f"Starting training with model: {model_type.upper()}")
        print(f"Use processed data: {use_processed}")
        if config_file:
            print(f"Using config file: {config_file}")
        print("="*60)
        
    else:
        # New style: python train.py --model timesformer --lr 0.001
        parser = argparse.ArgumentParser(description='Train video classification models')
        parser.add_argument('--model', type=str, default='timesformer', 
                           choices=['timesformer', 'vit', 'videomae'],
                           help='Model to train')
        parser.add_argument('--use-processed', type=str, default='true',
                           help='Use processed dataset (true/false)')
        parser.add_argument('--config', type=str, default=None,
                           help='Config file path')
        parser.add_argument('--experiment', type=str, default=None,
                           help='Experiment name from config file')
        parser.add_argument('--lr', '--learning-rate', type=float, default=None,
                           help='Learning rate override')
        parser.add_argument('--frames', type=int, default=None,
                           help='Number of frames override')
        parser.add_argument('--sampling-rate', type=int, default=None,
                           help='Sampling rate override')
        parser.add_argument('--batch-size', type=int, default=None,
                           help='Batch size override')
        parser.add_argument('--epochs', type=int, default=None,
                           help='Number of epochs override')
        parser.add_argument('--optimizer', type=str, default=None,
                           choices=['AdamW', 'Adam', 'SGD', 'RMSprop'],
                           help='Optimizer type override')
        parser.add_argument('--patience', type=int, default=None,
                           help='Early stopping patience (epochs). Set to 0 to disable early stopping')
        
        args = parser.parse_args()
        model_type = args.model
        use_processed = args.use_processed.lower() == 'true'
        config_file = args.config
        experiment_name = args.experiment
        learning_rate = args.lr
        num_frames = args.frames
        sampling_rate = args.sampling_rate
        batch_size = args.batch_size
        epochs = args.epochs
        optimizer_type = args.optimizer
        patience = args.patience
        
        print(f"Starting training with model: {model_type.upper()}")
        print(f"Use processed data: {use_processed}")
        if config_file:
            print(f"Using config file: {config_file}")
        if experiment_name:
            print(f"Using experiment: {experiment_name}")
        if learning_rate:
            print(f"Learning rate: {learning_rate}")
        if num_frames:
            print(f"Frames: {num_frames}")
        if sampling_rate:
            print(f"Sampling rate: {sampling_rate}")
        if optimizer_type:
            print(f"Optimizer: {optimizer_type}")
        print("="*60)
    
    try:
        model_path, best_accuracy, metrics = train_model(
            model_type=model_type, 
            use_processed=use_processed, 
            config_file=config_file,
            experiment_name=experiment_name,
            learning_rate=learning_rate,
            num_frames=num_frames,
            sampling_rate=sampling_rate,
            batch_size=batch_size,
            epochs=epochs,
            optimizer_type=optimizer_type,
            patience=patience
        )
        print("="*60)
        print(f"Training completed successfully!")
        print(f"Final Results:")
        print(f"   Model: {model_type.upper()}")
        print(f"   Best Accuracy: {best_accuracy:.3f}")
        print(f"   Model saved: {model_path}")
        if config_file:
            print(f"   Config used: {config_file}")
    except Exception as e:
        print(f"Training failed with error: {str(e)}")
        raise

# videomae_model.py
# VideoMAE video classification model

from transformers import AutoModelForVideoClassification, AutoConfig
import torch
import torch.nn as nn

class VideoMAEWrapper(nn.Module):
    """Wrapper for VideoMAE to handle 8-frame input by duplicating frames to 16."""
    
    def __init__(self, num_classes=24):
        super().__init__()
        # Load VideoMAE model configuration
        config = AutoConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
        config.num_labels = num_classes
        
        # Load VideoMAE model with new configuration
        self.videomae = AutoModelForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics",
            config=config,
            ignore_mismatched_sizes=True
        )
        
    def forward(self, pixel_values, labels=None, **kwargs):
        """
        Forward pass with frame duplication to handle 8->16 frame conversion.
        
        Args:
            pixel_values: Tensor of shape [batch_size, num_frames, channels, height, width]
                         Expected: [batch_size, 8, 3, 224, 224]
            labels: Optional labels for training
        """
        # VideoMAE expects 16 frames, we have 8 frames
        if pixel_values.shape[1] == 8:
            # Duplicate frames to get 16 frames: [B, 8, C, H, W] -> [B, 16, C, H, W]
            duplicated_frames = pixel_values.repeat_interleave(2, dim=1)
            print(f"VideoMAE: Duplicated frames from {pixel_values.shape} to {duplicated_frames.shape}")
        else:
            duplicated_frames = pixel_values
            
        # Forward through VideoMAE
        return self.videomae(pixel_values=duplicated_frames, labels=labels, **kwargs)

def get_videomae_model(num_classes=24):
    """Load VideoMAE pretrained model for video classification."""
    try:
        print(f"Loading VideoMAE model with {num_classes} classes...")
        
        # Use wrapper to handle frame duplication
        model = VideoMAEWrapper(num_classes=num_classes)
        
        print(f"VideoMAE model loaded successfully with {num_classes} classes")
        print("   Model configured for 8->16 frame conversion")
        return model
        
    except Exception as e:
        print(f"Error loading VideoMAE model: {e}")
        print("Trying TimeSFormer as backup...")
        
        try:
            # Fallback to TimeSFormer if VideoMAE fails
            config = AutoConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
            config.num_labels = num_classes
            
            model = AutoModelForVideoClassification.from_pretrained(
                "facebook/timesformer-base-finetuned-k400",
                config=config,
                ignore_mismatched_sizes=True
            )
            
            print(f"VideoMAE (TimeSFormer fallback) model loaded successfully with {num_classes} classes")
            return model
            
        except Exception as e2:
            raise Exception(f"Could not load any video model. VideoMAE: {e}, TimeSFormer: {e2}")

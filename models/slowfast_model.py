# slowfast_model.py
# SlowFast video classification model (using TimeSFormer as fallback)

from transformers import AutoModelForVideoClassification, AutoConfig
import torch
import torch.nn as nn

def get_slowfast_model(num_classes=24):
    """Load SlowFast (using TimeSFormer) model for video classification."""
    try:
        print(f"🔧 Loading video model with {num_classes} classes...")
        
        # Use TimeSFormer as it's more compatible with various input formats
        config = AutoConfig.from_pretrained("facebook/timesformer-base-finetuned-k400")
        config.num_labels = num_classes
        
        model = AutoModelForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400",
            config=config,
            ignore_mismatched_sizes=True
        )
        
        print(f"✅ SlowFast (TimeSFormer-based) model loaded successfully with {num_classes} classes")
        return model
        
    except Exception as e:
        print(f"❌ Error loading TimeSFormer model: {e}")
        print("🔄 Trying VideoMAE as backup...")
        
        try:
            # Fallback to VideoMAE
            config = AutoConfig.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics")
            config.num_labels = num_classes
            
            model = AutoModelForVideoClassification.from_pretrained(
                "MCG-NJU/videomae-base-finetuned-kinetics",
                config=config,
                ignore_mismatched_sizes=True
            )
            
            print(f"✅ SlowFast (VideoMAE-based) model loaded successfully with {num_classes} classes")
            return model
            
        except Exception as e2:
            raise Exception(f"Could not load any video model. TimeSFormer: {e}, VideoMAE: {e2}")

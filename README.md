<<<<<<< HEAD
# Video Action Recognition using Vision Transformer

A comprehensive action recognition system using Timesformer Vision Transformer for video classification on the HMDB-51 dataset.

## 🎯 Overview

This project implements a state-of-the-art video action recognition system using the Timesformer model, a Vision Transformer specifically designed for video understanding. The system can classify human actions in videos across 25 different action categories from the HMDB-51 dataset.

## 🚀 Features

- **Multiple Model Architectures**: Timesformer, ViT, VideoMAE
- **Comprehensive Training Pipeline**: Configurable hyperparameters, multiple optimizers
- **Advanced Evaluation Metrics**: Accuracy, confusion matrices, top-k analysis
- **Interactive Web Application**: Streamlit-based interface for real-time video analysis
- **Visualization Tools**: Loss curves, performance comparisons, confidence distributions

## 📁 Project Structure

```
Action_Recognition_using_VisionTransformer/
├── configs/
│   └── coursework_config.yaml          # Configuration file
├── models/
│   ├── timesformer_model.py            # Timesformer implementation
│   ├── vit_base_model.py               # ViT model for video
│   ├── vit_model.py                    # Model utilities
│   └── videomae_model.py               # VideoMAE implementation
├── src/
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation script
│   ├── dataset.py                      # Dataset handling
│   ├── transforms.py                   # Data transformations
│   └── utils.py                        # Utility functions
├── Results/                            # Training results and models
│   ├── lr - 0.001/SGD/                # Best performing model
│   ├── lr - 0.0005/                   # Other experiments
│   └── Default/                        # Default results
├── webapp.py                           # Streamlit web application
├── generate_loss_curves.py             # Loss visualization tool
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.9+

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/mohansree14/Action_Recognition-_using_VisionTransformer.git
cd Action_Recognition-_using_VisionTransformer
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Additional packages for the web app:
```bash
pip install streamlit opencv-python
```

## 📊 Dataset

This project uses the HMDB-51 dataset with 25 action categories:

- brush_hair, cartwheel, catch, chew, climb, climb_stairs
- draw_sword, eat, fencing, flic_flac, golf, handstand
- kiss, pick, pour, pullup, pushup, ride_bike
- shoot_bow, shoot_gun, situp, smile, smoke, throw, wave

## 🏋️ Training

### Basic Training
```bash
python src/train.py timesformer
```

### Advanced Training with Custom Parameters
```bash
python src/train.py timesformer --lr 0.001 --optimizer SGD --epochs 10 --frames 8 --stride 32
```

### Training Options
- **Models**: `timesformer`, `vit`, `videomae`
- **Optimizers**: `Adam`, `AdamW`, `SGD`, `RMSprop`
- **Learning Rates**: `0.0001`, `0.0005`, `0.001`

## 🔍 Evaluation

### Evaluate a Trained Model
```bash
python src/evaluate.py timesformer --model-path "Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.pth"
```

### Evaluate All Models
```bash
python src/evaluate.py all
```

## 📈 Results

### Best Performing Model
**Model**: Timesformer with SGD optimizer (LR=0.001)
- **Peak Validation Accuracy**: 100.0% (Epoch 4)
- **Final Validation Accuracy**: 96.1%
- **Model Path**: `Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.pth`

### Performance Comparison
| Optimizer | Learning Rate | Best Val Acc | Final Val Acc |
|-----------|--------------|--------------|---------------|
| SGD       | 0.001        | 100.0%       | 96.1%        |
| SGD       | 0.0005       | 75.8%        | 70.3%        |
| Adam      | 0.0005       | 14.2%        | 4.7%         |

## 🌐 Web Application

Launch the interactive web application for real-time video analysis:

```bash
streamlit run webapp.py
```

### Features:
- **Video Upload**: Support for MP4, AVI, MOV, MKV formats
- **Frame Extraction**: Configurable sampling methods
- **Real-time Prediction**: Action classification with confidence scores
- **Visualization**: Top-10 predictions, confidence distributions
- **Model Selection**: Choose from different trained models

### Web App Interface:
1. Load your trained model using the sidebar
2. Upload a video file
3. Configure frame extraction settings
4. Analyze the video for action recognition
5. View detailed results and confidence scores

## 📊 Visualization Tools

### Generate Loss Curves
```bash
python generate_loss_curves.py
```

Creates professional loss curve comparisons for different training configurations.

## ⚙️ Configuration

Edit `configs/coursework_config.yaml` to customize:

```yaml
dataset:
  root_dir: "path/to/dataset"
  categories: [list of action classes]

training:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 10
  
model:
  model_name: "facebook/timesformer-base-finetuned-k400"
  num_frames: 8
  stride: 32
```

## 🔧 Key Components

### Models
- **Timesformer**: State-of-the-art video transformer
- **ViT**: Vision Transformer adapted for video
- **VideoMAE**: Video Masked Autoencoder

### Training Pipeline
- Configurable hyperparameters
- Multiple optimizer support
- Automatic checkpointing
- Performance logging

### Evaluation Suite
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Top-k accuracy analysis
- Per-class performance analysis

## 📝 Usage Examples

### Training a Model
```python
# Train Timesformer with custom settings
python src/train.py timesformer --lr 0.001 --optimizer SGD --epochs 10
```

### Evaluating Performance
```python
# Evaluate specific model
python src/evaluate.py timesformer --model-path "path/to/model.pth"
```

### Using the Web App
```python
# Launch web interface
streamlit run webapp.py
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient accumulation

2. **Model Loading Errors**
   - Check model path exists
   - Verify class count matches training

3. **Web App Issues**
   - Ensure all dependencies installed
   - Check video format compatibility

## 📚 References

- [TimeSFormer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- [HMDB-51 Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohan Sree**
- GitHub: [@mohansree14](https://github.com/mohansree14)

## 🙏 Acknowledgments

- Facebook Research for the Timesformer model
- Hugging Face for the transformers library
- The creators of the HMDB-51 dataset

---

## 📊 Quick Start Guide

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Train a model**: `python src/train.py timesformer`
3. **Evaluate results**: `python src/evaluate.py timesformer`
4. **Launch web app**: `streamlit run webapp.py`

For detailed instructions, see the sections above! 🚀
=======
# Video Action Recognition using Vision Transformer

A comprehensive action recognition system using Timesformer Vision Transformer for video classification on the HMDB-51 dataset.

## 🎯 Overview

This project implements a state-of-the-art video action recognition system using the Timesformer model, a Vision Transformer specifically designed for video understanding. The system can classify human actions in videos across 25 different action categories from the HMDB-51 dataset.

## 🚀 Features

- **Multiple Model Architectures**: Timesformer, ViT, VideoMAE
- **Comprehensive Training Pipeline**: Configurable hyperparameters, multiple optimizers
- **Advanced Evaluation Metrics**: Accuracy, confusion matrices, top-k analysis
- **Interactive Web Application**: Streamlit-based interface for real-time video analysis
- **Visualization Tools**: Loss curves, performance comparisons, confidence distributions

## 📁 Project Structure

```
Action_Recognition_using_VisionTransformer/
├── configs/
│   └── coursework_config.yaml          # Configuration file
├── models/
│   ├── timesformer_model.py            # Timesformer implementation
│   ├── vit_base_model.py               # ViT model for video
│   ├── vit_model.py                    # Model utilities
│   └── videomae_model.py               # VideoMAE implementation
├── src/
│   ├── train.py                        # Training script
│   ├── evaluate.py                     # Evaluation script
│   ├── dataset.py                      # Dataset handling
│   ├── transforms.py                   # Data transformations
│   └── utils.py                        # Utility functions
├── Results/                            # Training results and models
│   ├── lr - 0.001/SGD/                # Best performing model
│   ├── lr - 0.0005/                   # Other experiments
│   └── Default/                        # Default results
├── webapp.py                           # Streamlit web application
├── generate_loss_curves.py             # Loss visualization tool
├── requirements.txt                    # Dependencies
└── README.md                           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch 1.9+

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/mohansree14/Action_Recognition-_using_VisionTransformer.git
cd Action_Recognition-_using_VisionTransformer
```

2. **Create virtual environment**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

Additional packages for the web app:
```bash
pip install streamlit opencv-python
```

## 📊 Dataset

This project uses the HMDB-51 dataset with 25 action categories:

- brush_hair, cartwheel, catch, chew, climb, climb_stairs
- draw_sword, eat, fencing, flic_flac, golf, handstand
- kiss, pick, pour, pullup, pushup, ride_bike
- shoot_bow, shoot_gun, situp, smile, smoke, throw, wave

## 🏋️ Training

### Basic Training
```bash
python src/train.py timesformer
```

### Advanced Training with Custom Parameters
```bash
python src/train.py timesformer --lr 0.001 --optimizer SGD --epochs 10 --frames 8 --stride 32
```

### Training Options
- **Models**: `timesformer`, `vit`, `videomae`
- **Optimizers**: `Adam`, `AdamW`, `SGD`, `RMSprop`
- **Learning Rates**: `0.0001`, `0.0005`, `0.001`

## 🔍 Evaluation

### Evaluate a Trained Model
```bash
python src/evaluate.py timesformer --model-path "Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.pth"
```

### Evaluate All Models
```bash
python src/evaluate.py all
```

## 📈 Results

### Best Performing Model
**Model**: Timesformer with SGD optimizer (LR=0.001)
- **Peak Validation Accuracy**: 100.0% (Epoch 4)
- **Final Validation Accuracy**: 96.1%
- **Model Path**: `Results/lr - 0.001/SGD/timesformer_lr000100_f8_s32_20250805_134346.pth`

### Performance Comparison
| Optimizer | Learning Rate | Best Val Acc | Final Val Acc |
|-----------|--------------|--------------|---------------|
| SGD       | 0.001        | 100.0%       | 96.1%        |
| SGD       | 0.0005       | 75.8%        | 70.3%        |
| Adam      | 0.0005       | 14.2%        | 4.7%         |

## 🌐 Web Application

Launch the interactive web application for real-time video analysis:

```bash
streamlit run webapp.py
```

### Features:
- **Video Upload**: Support for MP4, AVI, MOV, MKV formats
- **Frame Extraction**: Configurable sampling methods
- **Real-time Prediction**: Action classification with confidence scores
- **Visualization**: Top-10 predictions, confidence distributions
- **Model Selection**: Choose from different trained models

### Web App Interface:
1. Load your trained model using the sidebar
2. Upload a video file
3. Configure frame extraction settings
4. Analyze the video for action recognition
5. View detailed results and confidence scores

## 📊 Visualization Tools

### Generate Loss Curves
```bash
python generate_loss_curves.py
```

Creates professional loss curve comparisons for different training configurations.

## ⚙️ Configuration

Edit `configs/coursework_config.yaml` to customize:

```yaml
dataset:
  root_dir: "path/to/dataset"
  categories: [list of action classes]

training:
  batch_size: 4
  learning_rate: 0.001
  num_epochs: 10
  
model:
  model_name: "facebook/timesformer-base-finetuned-k400"
  num_frames: 8
  stride: 32
```

## 🔧 Key Components

### Models
- **Timesformer**: State-of-the-art video transformer
- **ViT**: Vision Transformer adapted for video
- **VideoMAE**: Video Masked Autoencoder

### Training Pipeline
- Configurable hyperparameters
- Multiple optimizer support
- Automatic checkpointing
- Performance logging

### Evaluation Suite
- Comprehensive metrics (accuracy, precision, recall, F1)
- Confusion matrices
- Top-k accuracy analysis
- Per-class performance analysis

## 📝 Usage Examples

### Training a Model
```python
# Train Timesformer with custom settings
python src/train.py timesformer --lr 0.001 --optimizer SGD --epochs 10
```

### Evaluating Performance
```python
# Evaluate specific model
python src/evaluate.py timesformer --model-path "path/to/model.pth"
```

### Using the Web App
```python
# Launch web interface
streamlit run webapp.py
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce batch size in config
   - Use gradient accumulation

2. **Model Loading Errors**
   - Check model path exists
   - Verify class count matches training

3. **Web App Issues**
   - Ensure all dependencies installed
   - Check video format compatibility

## 📚 References

- [TimeSFormer: Is Space-Time Attention All You Need for Video Understanding?](https://arxiv.org/abs/2102.05095)
- [HMDB-51 Dataset](https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/)
- [Hugging Face Transformers](https://huggingface.co/transformers/)

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

**Mohan Sree**
- GitHub: [@mohansree14](https://github.com/mohansree14)

## 🙏 Acknowledgments

- Facebook Research for the Timesformer model
- Hugging Face for the transformers library
- The creators of the HMDB-51 dataset

---

## 📊 Quick Start Guide

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Train a model**: `python src/train.py timesformer`
3. **Evaluate results**: `python src/evaluate.py timesformer`
4. **Launch web app**: `streamlit run webapp.py`

For detailed instructions, see the sections above! 🚀
>>>>>>> 28d1049ffd1c191af8331bacb7e4645904ca6e5a

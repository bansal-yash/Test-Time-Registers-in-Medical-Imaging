# Test Time Registers in Medical Imaging

A research project investigating the behavior and impact of register tokens in Vision Transformers (CLIP and DINOv2) for medical imaging classification tasks.

## Project Overview

This project systematically investigates register token detection and removal across CLIP and DINOv2 architectures on three diverse medical imaging datasets. The research explores whether naturally emerged register-like tokens in pre-trained models improve or degrade classification performance, and how they affect attention patterns and model interpretability.

**This project is based on the GitHub repository**: https://github.com/nickjiang2378/test-time-registers


## Datasets

The project evaluates three medical imaging datasets:

1. **Pacemaker Dataset**: Pacemaker device classification (45 classes)
2. **OrthoNet Dataset**: Orthopedic implant classification (12 classes)
3. **APTOS Dataset**: Diabetic retinopathy detection (5 classes)

## Models

Two Vision Transformer architectures are investigated:

- **CLIP (ViT-B/16)**: Pre-trained on LAION-2B dataset
- **DINOv2 (ViT-L/14)**: Self-supervised learning with self-distillation

## Key Features

### Register Token Detection
- Analyzes patch-level output norms across layers
- Identifies neurons with consistently high activation norms
- Applies sparsity filtering to distinguish register tokens
- Ranks register neurons by consistency score

### Test-Time Register (TTR) Intervention
- **Baseline (no-ttr)**: Standard model inference
- **Ablated (ttr)**: Register neurons zeroed out, explicit register tokens appended
- Implemented via activation hooks for fine-grained control

### Evaluation Metrics
- Top-1 and Top-3 accuracy
- Macro-averaged F1 score, precision, and recall
- Multi-class AUC-ROC
- Attention pattern visualization

## Usage

### Testing Models
```bash
# Test with CLIP
python test_pacemaker.py --model clip --image_path /path/to/pacemaker-data
python test_orthonet.py --model clip --image_path /path/to/orthonet-data
python test_aptos.py --model clip --image_path /path/to/aptos-data

# Test with DINOv2
python test_pacemaker.py --model dinov2 --image_path /path/to/pacemaker-data
python test_orthonet.py --model dinov2 --image_path /path/to/orthonet-data
python test_aptos.py --model dinov2 --image_path /path/to/aptos-data
```

### Configuration

Models can be configured by modifying the config dictionaries in the test scripts:
```python
# CLIP configuration
config = {
    "model_name": "ViT-B-16",
    "pretrained": "laion2b_s34b_b88k",
    "device": "mps",  # or "cuda" or "cpu"
    "highest_layer": 5,
    "detect_outliers_layer": -1,
    "register_norm_threshold": 30,
    "top_k": 20,
}

# DINOv2 configuration
config = {
    "backbone_size": "vitl14",
    "device": "mps",
    "detect_outliers_layer": -2,
    "register_norm_threshold": 150,
    "highest_layer": 19,
    "top_k": 50,
}
```

## Implementation Details

### Hook Manager System
- Captures attention maps from all layers
- Records layer outputs
- Intervenes on specific neurons during forward pass
- Injects register tokens at specified positions

### Dataset Processing
- Custom PyTorch Dataset classes for each dataset
- Generates and caches feature representations
- Supports both baseline and register-ablated inference modes
- Efficient in-memory representation storage

### Classifier Training
- Linear probe on frozen ViT backbone
- AdamW optimizer with cosine annealing schedule
- Early stopping based on validation loss
- Gradient clipping for stable training

## Key Findings

1. **Register tokens exist naturally**: Both CLIP and DINOv2 exhibit register-like behavior without explicit training
2. **Architecture-dependent severity**: Different models show varying degrees of artifact emergence
3. **Task-dependent performance**: Impact varies significantly by dataset and model
4. **Improved interpretability**: Register removal consistently produces cleaner attention patterns
5. **Functional roles**: Naturally emerged registers may serve context-dependent purposes

## References

1. Darcet, T., et al. (2023). "Vision Transformers Need Registers." arXiv:2309.16588
2. Walmer, M., et al. (2024). "Teaching Matters: Investigating the Role of Supervision in Vision Transformers." CVPR
3. Shaharabany, T., et al. (2023). "PROMPTCAM: Bringing Prompts to Image Segmentation and Visual Explanation." arXiv:2310.12955

## Authors

- Spandan Kukade (2022CS51138)
- Yash Bansal (2022CS51133)

**Course**: COL828 Advanced Computer Vision  
**Supervisor**: Prof. Chetan Arora  
**Department**: Computer Science and Engineering   
**Institution**: Indian Institute of Technology Delhi  
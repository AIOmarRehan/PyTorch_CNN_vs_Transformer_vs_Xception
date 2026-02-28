[If you would like a detailed explanation of this project, please refer to the Medium article below.](https://medium.com/@ai.omar.rehan/comparing-three-computer-vision-approaches-in-pytorch-from-custom-cnns-to-advanced-transfer-ea3f89f61ed4)

---
[The project is also available for testing on Hugging Face.](https://huggingface.co/spaces/AIOmarRehan/CV_Model_Comparison_in_PyTorch)

# PyTorch Model Comparison: From Custom CNNs to Advanced Transfer Learning

![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat\&logo=python\&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat\&logo=pytorch\&logoColor=white)
![Gradio](https://img.shields.io/badge/Gradio-4.0+-FF6F00?style=flat\&logo=gradio\&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat\&logo=huggingface\&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=flat\&logo=docker\&logoColor=white)

---

## Overview

This project compares **three computer vision approaches in PyTorch** on a vehicle classification task:

1. Custom CNN (trained from scratch)
2. Vision Transformer (DeiT-Tiny)
3. Xception with two-phase transfer learning

The goal is to answer a practical question:

> On small or moderately sized datasets, should you train from scratch or use transfer learning?

The results clearly show that **transfer learning dramatically improves generalization and reliability**, especially when data and compute are limited.

---

## Architectures Compared

### Custom CNN (From Scratch)

A traditional convolutional network built manually with Conv → ReLU → Pooling blocks and fully connected layers.

**Philosophy:** Full architectural control, no pre-training.

Minimal structure:

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(64 * 56 * 56, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
```

**Reality on small datasets:**

* Slower convergence
* Higher variance
* Larger generalization gap

---

### Vision Transformer (DeiT-Tiny)

Using Hugging Face's pre-trained Vision Transformer:

```python
model = AutoModelForImageClassification.from_pretrained(
    "facebook/deit-tiny-patch16-224",
    num_labels=num_classes,
    ignore_mismatched_sizes=True
)
```

Trained with the Hugging Face `Trainer` API.

**Advantages:**

* Stable convergence
* Lightweight
* Easy deployment
* Good performance-to-efficiency ratio

---

### Xception (Two-Phase Transfer Learning)

Implemented using `timm`.

### Phase 1 - Train Classifier Head Only

```python
model = timm.create_model("xception", pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(
    nn.Linear(in_features, 512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, num_classes)
)
```

### Phase 2 - Fine-Tune Selected Layers

```python
for name, param in model.named_parameters():
    if "block14" in name or "fc" in name:
        param.requires_grad = True
```

Lower learning rate used during fine-tuning.

**Result:**
- Smoothest training curves
- Lowest validation loss
- Highest test accuracy
- Strongest performance on unseen internet images

---

## Comparative Results

| Model      | Validation Performance | Generalization | Stability   |
| ---------- | ---------------------- | -------------- | ----------- |
| Custom CNN | High variance          | Weak           | Unstable    |
| DeiT-Tiny  | Strong                 | Good           | Stable      |
| Xception   | Best                   | Excellent      | Very Stable |

### Key Insight

> High validation accuracy does NOT guarantee real-world reliability.

Custom CNN achieved strong validation scores (~87%) but struggled more on distribution shifts.

Xception consistently generalized better.

---

## Experimental Visualizations

### Dataset Distribution Across All Three Models:

![Chart](https://files.catbox.moe/eyuftl.png)

---

### Xception Model:
![Accuracy & Loss](https://files.catbox.moe/qv7n6e.png)
### Custom CNN Model:
![Accuracy & Loss](https://files.catbox.moe/ch8s5d.png)

---

### Confusion Matrix between both Models:

| **Custom CNN** | **Xception** |
|------------|----------|
| <img src="https://files.catbox.moe/aulaxo.webp" width="100%"> | <img src="https://files.catbox.moe/gy6yno.webp" width="100%"> |

---

## Example Test Results (Custom CNN)

```
Test Accuracy: 87.89%

Macro Avg:
Precision: 0.8852
Recall:    0.8794
F1-Score:  0.8789
```

Despite solid metrics, performance dropped more noticeably on unseen real-world images compared to Xception.

---

## Deployment

### Run Locally

```bash
pip install -r requirements.txt
python app.py
```

Access at:

```
http://localhost:7860
```

---

## When to Use Each Approach

### Use Custom CNN if:

* Domain is highly specialized
* Pre-trained features don’t apply
* You need full architectural control

### Use Transfer Learning (e.g. DeiT or Xception) if:

* You want fast experimentation
* Efficiency matters
* You prefer high-level APIs
* You want best accuracy
* You care about generalization
* You need production-grade reliability

---

## Final Conclusion

On small or moderately sized datasets:

> Transfer learning isn’t an optimization - it’s a necessity.

Training from scratch forces the model to learn both general visual features and task-specific knowledge simultaneously.

Pre-trained models already understand edges, textures, and spatial structure.
Your dataset only needs to teach classification boundaries.

For most real-world tasks:

* Start with transfer learning
* Fine-tune carefully
* Only train from scratch if absolutely necessary

---

## Results

<p align="center">
  <a href="https://files.catbox.moe/ss5ohr.mp4">
    <img src="https://files.catbox.moe/3x5mp7.webp" width="400">
  </a>
</p>

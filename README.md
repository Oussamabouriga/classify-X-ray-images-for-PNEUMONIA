# 🩻 Pneumonia Detection from Chest X-Rays using Deep Learning

Pneumonia is a serious respiratory illness that requires timely and accurate diagnosis for effective treatment. In this project, a deep learning model is fine-tuned to assist with diagnosing pneumonia by analyzing chest X-ray images.

By leveraging a pre-trained **ResNet-18** model, the goal is to classify X-ray images into two categories:
- **NORMAL** — healthy lungs
- **PNEUMONIA** — lungs affected by pneumonia

This project demonstrates how transfer learning and convolutional neural networks can be used to build a reliable diagnostic tool with limited data.

---

## 📁 Dataset

The dataset used contains preprocessed chest X-ray images organized into:
- `train/` — 150 images per class (NORMAL & PNEUMONIA)
- `test/` — 50 images per class (NORMAL & PNEUMONIA)

Total:
- **300 training images**
- **100 testing images**

The dataset is automatically loaded using PyTorch's `ImageFolder` and `DataLoader`.

---

## 🚀 Model & Training

- Pre-trained **ResNet-18** from `torchvision.models`
- Only the final fully connected layer (`resnet18.fc`) is trained (rest is frozen)
- Binary classification using `BCEWithLogitsLoss`
- Optimized using `Adam` with learning rate = `0.01`
- Training enhanced with:
  - Accuracy & F1 score tracking
  - Learning rate scheduler
  - Gradient clipping

---

## 🔍 Evaluation

Model performance is evaluated using:
- **Accuracy**
- **F1 Score**
- Visual inspection of predictions on test images

The notebook also supports:
- **Balanced visualization** of predictions (equal examples from each class)
- **Inference on external images** placed in a custom folder (`images_form_other_dataset`)

---

## 🖼️ Custom Prediction

To test the model on your own X-ray images:
1. Place them in the `images_form_other_dataset/` folder.
2. Name files like `normal1.png`, `pneumonia1.png` to reflect their true label.
3. Run the provided script to visualize predictions vs actual labels.

---

## 🧪 Example Results

![Sample X-ray results](x-rays_sample.png)

---

## 📦 Requirements

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- torchmetrics
- PIL (Pillow)

Install dependencies:

```bash
pip install torch torchvision matplotlib torchmetrics pillow
```

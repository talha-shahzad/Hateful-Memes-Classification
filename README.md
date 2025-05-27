
# Hateful Memes Classification

This project implements multimodal classification of hateful memes using image and text inputs. We use **late fusion** and **early fusion** strategies to compare performance using models like ResNet, CNN, BERT, and LSTM.

## 🧠 Introduction

This project implements a **multimodal classification system** for the Hateful Memes dataset, combining both visual and textual features. The task involves predicting whether a meme is **hateful** or **non-hateful** by analyzing its image and associated text.

We design and compare two multimodal fusion strategies:

---

### 🔹 Late Fusion

Late Fusion processes the image and text inputs independently using separate encoders. The outputs from these encoders are then **concatenated at a later stage**, and passed through a classifier to make the final prediction.

- **Image encoders**:  
  - `ResNet-50` (pretrained on ImageNet)  
  - Custom `CNN` with 3-layer structure  

- **Text encoders**:  
  - `BERT-base` (transformer-based encoder)  
  - `LSTM` (Bi-directional recurrent network with GloVe-style embeddings)  

#### 🧪 Model Variants:
- ResNet + BERT (Late)
- ResNet + LSTM (Late)
- CNN + BERT (Late)
- CNN + LSTM (Late)

---

### 🔹 Early Fusion

In Early Fusion, image and text features are extracted and **combined at an earlier stage**—before classification—allowing the model to learn **joint representations** from the start.

- The same encoders are used (ResNet/CNN for images, BERT/LSTM for text).
- The image and text embeddings are fused right after encoding, and passed to a shared classifier.

#### 🧪 Model Variants:
- ResNet + BERT (Early)
- ResNet + LSTM (Early)
- CNN + BERT (Early)
- CNN + LSTM (Early)

---

Each variant is trained and evaluated with:
- **Cross-entropy loss**
- **Adam optimizer**
- **Batch size:** configurable (commonly 32)
- **Metrics tracked:** AUROC, Precision, Recall, F1-score

Both fusion strategies are compared in terms of **performance** and **computational tradeoffs**, with full training logs and evaluation visualizations stored via **TensorBoard** and **matplotlib**.


---

## 📁 Folder Structure

The dataset folder should look like this after setup:

```
data/
├── img/
│   ├── 00001.png
│   ├── ...
├── train.jsonl
├── dev.jsonl
├── test.jsonl
```

Each JSONL file contains meme samples with fields like `text`, `img` (path), and `label`.

---

## 🔧 Google Colab Setup

1. **Mount Google Drive:**

```python
from google.colab import drive

# Mount your Google Drive
def mount_drive():
    drive.mount('/content/drive')

mount_drive()
```

2. **Upload Dataset:**

Place the unzipped dataset inside your Google Drive, e.g., `/MyDrive/data`.

3. **Set Dataset Path in Code:**

```python
data_root = '/content/drive/MyDrive/data'
```

Make sure `data_root` contains the expected folder structure as shown above.

---

## ⚡ Lightning AI Setup

1. **Install KaggleHub:**

```bash
pip install kagglehub
```

2. **Download Dataset Automatically:**

```python
import kagglehub

# Download and unpack the dataset
path = kagglehub.dataset_download("marafey/hateful-memes-dataset")
data_root = path + '/data'
print("Path to dataset files:", data_root)
```

This will download the dataset to the Lightning AI instance under `/data`.

---

## 💻 Hardware Requirements

- **GPU:** T4 recommended (16GB VRAM). You can select this in:
  - Google Colab: `Runtime > Change runtime type > GPU > T4`
  - Lightning AI Studio: Use "GPU (T4)" when launching an instance

---

## 📊 Logging

- Training logs are stored using TensorBoard in:
  - Google Colab: `/content/drive/MyDrive/logs`
  - Lightning AI: Use local or `wandb` as needed

Launch TensorBoard in Colab:

```python
%load_ext tensorboard
%tensorboard --logdir /content/drive/MyDrive/logs
```

---

## 📈 Evaluation

- Metrics: AUROC, Precision, Recall, F1
- Visuals: Confusion Matrix, ROC Curve
- Sample analysis: View correctly/incorrectly classified memes

---

## 📦 Requirements

Install required packages:

```bash
pip install torch torchvision transformers nltk scikit-learn wordcloud matplotlib kagglehub
```

---

## ✅ Done!

You're now ready to explore fusion strategies and evaluate meme classification models!

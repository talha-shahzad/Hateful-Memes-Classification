
# Hateful Memes Classification

This project implements multimodal classification of hateful memes using image and text inputs. We use **late fusion** and **early fusion** strategies to compare performance using models like ResNet, CNN, BERT, and LSTM.

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

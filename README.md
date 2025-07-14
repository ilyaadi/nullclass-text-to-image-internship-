# 🧠 NullClass Internship: Text-to-Image Generation Tasks

This repository contains three internship tasks completed as part of the NullClass Data Science Internship (July 8 – August 8, 2025). Each task showcases practical implementations in image processing, text embedding, and deep learning using GANs with attention mechanisms.

---

## 📂 Project Structure

|-- Task-1__Image_Display
    |-- requirements.txt
    |-- smileyFace.jpg
    |-- Task_1.ipynb
|-- Task-2_Text_Preprocessing
    |-- requirements.txt
    |-- text_tokenization.ipynb
|-- Task-3_Text_To_Image_model
    |-- attention_gan.ipynb
    |-- requirements.txt
    
---

## ✅ Task Overview

### 🖼 Task 1: Image Loading and Visualization
- Uses **OpenCV** and **Matplotlib** to read and display an image.
- Demonstrates fundamental image processing.
- 📍 Notebook: `Task_1.ipynb`

---

### ✍️ Task 2: Text Tokenization and Embedding
- Uses **Hugging Face Transformers** (`bert-base-uncased`) to:
  - Tokenize a text input
  - Generate contextual embeddings
- 📍 Notebook: `text_tokenization.ipynb`

---

### 🧬 Task 3: GAN with Cross-Attention for Text-to-Image
- Implements a **Generator model** using PyTorch.
- Integrates a **Cross-Attention** module that uses text embeddings to guide image generation.
- 💾 **Model Weights**:  
  [Download `generator_model_weights.pth`](https://drive.google.com/file/d/1tdL2D7qGrRdgEkC59BbgJEYhkxNGXWd1/view?usp=drive_link)
- 📍 Notebook: `attention_gan.ipynb`

---

## 🔧 Setup Instructions

Each task folder contains its own `requirements.txt`.

### ✅ Install all requirements at once:
```bash
pip install torch torchvision transformers matplotlib opencv-python
```

### 🔍 Or install task-wise:
```bash
# Task 1
pip install -r Task-1__Image_Display/requirements.txt

# Task 2
pip install -r Task-2_Text_Preprocessing/requirements.txt

# Task 3
pip install -r Task-3_Text_To_Image_model/requirements.txt
```

---

## 🧪 Sample Usage – Loading Trained Generator Weights

```python
from attention_gan import Generator
import torch

gen = Generator()
gen.load_state_dict(torch.load("generator_model_weights.pth", map_location='cpu'))
gen.eval()

noise = torch.randn(1, 100)
embedding = torch.randn(1, 768)
generated_image = gen(noise, embedding)

print(generated_image.shape)  # torch.Size([1, 3, 64, 64])
```

---

## 📃 Internship Summary

- 🗓 **Period**: July 8 – August 8, 2025  
- 🏢 **Organization**: [NullClass](https://nullclass.com/)  
- 🧠 **Focus**: Text-to-Image Generation using Deep Learning  
- 🛠 **Tools Used**: Python, PyTorch, Transformers, OpenCV, Matplotlib

---

## ✅ Completion Status

- [x] Task 1 – Image Display  
- [x] Task 2 – Text Embedding  
- [x] Task 3 – GAN with Attention  
- [x] Model Weights Saved and Linked  
- [x] All Notebooks Completed  
- [x] Requirements Included

---

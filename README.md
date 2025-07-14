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

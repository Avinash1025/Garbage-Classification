# 🗑️ Garbage Classification using EfficientNet (Streamlit App)

A deep learning-based image classification project that detects and classifies garbage images into one of six categories (Cardboard, Glass, Metal, Paper, Plastic, Trash) using the **EfficientNet** architecture and a Streamlit-based web interface.

---

## 🚀 Learning Objectives

- Understand how to implement image classification using transfer learning
- Work with custom datasets and PyTorch Lightning
- Deploy deep learning models using **Streamlit**

---

## 🧰 Tools & Technologies Used

- Python 🐍
- PyTorch & PyTorch Lightning
- EfficientNet
- Torchvision
- PIL
- Streamlit
- Jupyter Notebook
- Scikit-learn

---

## 🧠 Problem Statement

In many urban environments, improper segregation of waste leads to poor recycling efficiency and increased pollution. Manual sorting is time-consuming and inefficient. This project automates garbage classification to aid smart waste management systems.

---

## ✅ Solution

We used **EfficientNet**, a powerful CNN model, trained on garbage image data, and deployed it as a Streamlit web app for real-time predictions.

The model classifies images into the following six classes:
- **Cardboard**
- **Glass**
- **Metal**
- **Paper**
- **Plastic**
- **Trash**


---

## 🔍 Methodology

1. **Data Preparation**: Organized images into `train`, `val`, `test` folders.
2. **Model**: Used EfficientNet with transfer learning (custom `EfficientLite` class).
3. **Training**: Trained with PyTorch Lightning on 6-class garbage dataset.
4. **Evaluation**: Achieved >65% validation accuracy.
5. **Deployment**: Built a user-friendly UI using Streamlit to upload and classify garbage images.


---

## ✅ How to Run Locally

```bash
git clone https://github.com/yourusername/garbage-classification.git
cd garbage-classification
pip install -r requirements.txt
streamlit run app.py
```

---

## 🙋‍♂️ Author
Avinash Gour
Final Year B.Tech, Mathematics and Computing
MITS Gwalior | 2025
LinkedIn • GitHub



# 🫀 Myo AI: A Multimodal Fusion Framework for Cardiovascular Risk Stratification

> **Byte 2 Beat Hackathon (Hack4Health 2026) — Technical Track Submission**

**Myo AI** is an advanced, multimodal machine learning system designed to bridge the gap between clinical history and physiological signals. Unlike traditional singleton models, Myo AI orchestrates a **"Battle Royale" Tournament** among five independent architectures—ranging from Probabilistic Naive Bayes to Deep Learning CNNs—to identify the optimal strategy for cardiac risk prediction.

## 🚀 Try it Out
Click the badge below to run the full pipeline, witness the model tournament, and interact with the **Bio-Deck Dashboard** directly in your browser using Google Colab's free GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4hmed-n/Myo-AI/blob/main/Myo_AI.ipynb)

*Note: The system automatically ingests and harmonizes over 600MB of medical data. Initial setup may take 1-2 minutes.*

---

## 📸 Interface & Capabilities

### 🧬 The Chronos Time-Travel Dashboard
*An interactive "Digital Twin" simulator that projects patient risk over the next 20 years.*
![Myo-Sim Dashboard](https://github.com/4hmed-n/Myo-AI/blob/main/assets/dashboard_preview.png?raw=true)
*(Replace the link above with your actual screenshot of the dashboard)*

### 🔍 The Oracle Layer (Explainability)
*Granular SHAP force plots explaining exactly why a patient was flagged.*
![SHAP Visualization](https://github.com/4hmed-n/Myo-AI/blob/main/assets/shap_preview.png?raw=true)
*(Replace with your SHAP screenshot)*

---

## 🛠️ Built With

### **Core Stack**
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras_CNN-FF6F00?style=flat&logo=tensorflow&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-Tournament_Engine-F7931E?style=flat&logo=scikit-learn&logoColor=white)

### **Data & Engineering**
* **Pandas & NumPy:** High-performance tensor manipulation.
* **SciPy:** Statistical signal extraction (Kurtosis/Skewness) from raw ECGs.
* **Gdown:** Automated secure ingestion from Google Drive.

### **Intelligence & UI**
* **SHAP (Oracle Layer):** Game-theoretic feature explainability.
* **IPyWidgets (Bio-Deck):** Interactive real-time risk simulation.
* **Matplotlib & Seaborn:** Diagnostic visualizations (ROC Curves, Confusion Matrices).

---

## 🧠 System Architecture

Myo AI operates on a vertically scalable **3-Layer Stack**:

### **Layer 1: The Foundation (Data Engineering)**
* **Synapse Ingestion Engine:** Harmonizes 4 disparate datasets into a massive cohort of **140,918** patients.
* **Pulse Harmonization Engine:** A streaming algorithm that processes **600MB+** of raw ECG waveforms in memory-efficient chunks to extract physiological fingerprints.
* **Catalyst Feature Synthesizer:** Fuses clinical vitals with signal data, engineering biomarkers like *Pulse Pressure* and *BMI* while handling sensor missingness.

### **Layer 2: The Tournament (Model Selection)**
Five fully independent KDD pipelines compete in a stratified validation environment. The system automatically selects the champion based on ROC-AUC performance:
1.  🛡️ **Aegis Protocol:** Random Forest Classifier
2.  ⚡ **Myo-Core Engine:** Histogram Gradient Boosting
3.  👁️ **Sentinel Node:** Gaussian Naive Bayes
4.  🛡️ **Vanguard System:** Logistic Regression
5.  💓 **Pulse-Sync:** 1D-Convolutional Neural Network (Deep Learning)

### **Layer 3: The Intelligence (Analysis)**
* **Oracle Layer:** Explains "Black Box" decisions using SHAP values.
* **Zenith Map:** Unsupervised clustering (PCA + K-Means) to find hidden patient risk phenotypes.
* **Chronos Engine:** A predictive simulator for longitudinal risk projection.

---

## 🏆 Tournament Results
The system evaluated all candidates on a held-out test set of **28,184 patients**.

| Rank | Model Architecture | Accuracy | ROC-AUC | Status |
| :--- | :--- | :--- | :--- | :--- |
| 🥇 | **Aegis Protocol (Random Forest)** | **73.79%** | **0.8082** | **CHAMPION** |
| 🥈 | Myo-Core (Gradient Boosting) | 73.77% | 0.8080 | Runner-up |
| 🥉 | Pulse-Sync (1D-CNN) | 73.19% | 0.7949 | Deep Learning |
| 4 | Vanguard System (LogReg) | 72.21% | 0.7758 | Baseline |
| 5 | Sentinel Node (Naive Bayes) | 50.41% | 0.5075 | Baseline |

*Key Insight: The **Aegis Protocol** demonstrated that robust ensemble methods can outperform complex Deep Learning architectures on structured clinical data, achieving the highest sensitivity with significantly lower training overhead.*

---

## 📜 License
Distributed under the MIT License. See `LICENSE` for more information.

---
*Created by **Muhammad Ahmed**.*

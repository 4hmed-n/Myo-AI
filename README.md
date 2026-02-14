# Myo AI: A Multimodal Fusion Framework for Cardiovascular Risk Stratification

> **Byte 2 Beat Hackathon (Hack4Health 2026) — Technical Track Submission**

**Myo AI** is an advanced, multimodal machine learning system designed to bridge the gap between clinical history and physiological signals. Unlike traditional singleton models, Myo AI orchestrates a **"Battle Royale" Tournament** among five independent architectures—ranging from Probabilistic Naive Bayes to Deep Learning CNNs—to identify the optimal strategy for cardiac risk prediction.

## 🚀 Try it Out
Click the badge below to run the full pipeline, witness the model tournament, and interact with the **Bio-Deck Dashboard** directly in your browser using Google Colab's free GPU.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/4hmed-n/Myo-AI/blob/main/Myo%20AI.ipynb)

*Note: The system automatically ingests and harmonizes over 600MB of medical data. Initial setup may take 1-2 minutes.*

---

## 📸 Interface & Capabilities

### 🧬 The Chronos Time-Travel Dashboard
*An interactive "Digital Twin" simulator that projects patient risk over the next 20 years.*
![Myo-Sim Dashboard](https://github.com/4hmed-n/Myo-AI/blob/main/assets/Patient%20Simulator%20UI.png?raw=true)

### 🔍 The Oracle Layer (Explainability)
*Granular SHAP waterfall plots explaining exactly why a specific patient was flagged.*
![SHAP Waterfall](https://github.com/4hmed-n/Myo-AI/blob/main/assets/Waterfall%20SHAP.png?raw=true)

---

## 📊 Performance & Validation

### Tournament Leaderboard (ROC-AUC)
*The system compares 5 models simultaneously. The tree-based ensembles (RF & HGB) dominated the Deep Learning CNN.*
![ROC Curve](https://github.com/4hmed-n/Myo-AI/blob/main/assets/ROC-AUC.png?raw=true)

### Confusion Matrix Grid
*Visualizing the classification fidelity of all contestants.*
![Confusion Matrix](https://github.com/4hmed-n/Myo-AI/blob/main/assets/CF.png?raw=true)

### Unsupervised Risk Stratification (The Zenith Map)
*PCA + K-Means clustering identifying distinct "Risk Phenotypes" (Low/Medium/High) entirely unsupervised.*
![Zenith Map](https://github.com/4hmed-n/Myo-AI/blob/main/assets/Scatter%20Plot.png?raw=true)

---

## 🧠 Feature Intelligence

### Global Risk Drivers (Beeswarm Plot)
*Shows the directionality of risk. High Blood Pressure (Red) pushes risk to the right (positive), while Physical Activity (Blue) lowers it.*
![Beeswarm Plot](https://github.com/4hmed-n/Myo-AI/blob/main/assets/Beeswarm%20SHAP.png?raw=true)

### Feature Magnitude (Mean SHAP)
*A consolidated ranking of feature importance by absolute impact, providing a 'consensus' view of what drives the Aegis Protocol's decisions.*
![Mean SHAP](https://github.com/4hmed-n/Myo-AI/blob/main/assets/Mean%20SHAP%20FI.png?raw=true)


### Statistical Significance (Power SHAP)
*Using PowerSHAP to filter out noise. This ensures every feature utilized by Myo AI has a predictive power significantly higher than a random uniform distribution.*
![Power SHAP](https://github.com/4hmed-n/Myo-AI/blob/main/assets/Power%20SHAP.png?raw=true)


### Permutation Importance
*Ranking features by how much the model degrades when they are shuffled. Systolic BP (`ap_hi`) is the #1 predictor.*
![Permutation Importance](https://github.com/4hmed-n/Myo-AI/blob/main/assets/PI%20Plot.png?raw=true)

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
* **PowerSHAP:** Feature selection via statistical shadow variables.
* **IPyWidgets (Bio-Deck):** Interactive real-time risk simulation.
* **Matplotlib & Seaborn:** Diagnostic visualizations.

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

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---
*Created by **Muhammad Ahmed**.*

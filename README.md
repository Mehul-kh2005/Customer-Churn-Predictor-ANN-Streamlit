# 🔁 Customer Churn Prediction using Deep Learning (ANN)

## 📊 Project Overview

This project predicts customer churn for a bank using an **Artificial Neural Network (ANN)** built in **TensorFlow/Keras**. The goal is to identify customers likely to leave the bank so the business can take proactive retention measures.

This is a full machine learning pipeline project with:

- ✅ Cleaned & encoded raw customer data  
- ✅ ANN model built, trained, and evaluated with multiple metrics  
- ✅ Evaluation includes **ROC-AUC**, **precision**, **recall**, **F1-score**, and **confusion matrix**  
- ✅ Training is optimized using **early stopping** and **TensorBoard logging**  
- ✅ Final model is exported and ready for deployment  

---

## 🚀 Key Features

- ✅ End-to-end ML workflow: **preprocessing → training → evaluation → saving**  
- ✅ `StandardScaler`, `LabelEncoder`, and `OneHotEncoder` with saved encoders  
- ✅ Built using **TensorFlow Sequential API**  
- ✅ Metrics: **Accuracy**, **ROC-AUC**, **Precision**, **Recall**, **F1-score**  
- ✅ Model checkpointing using **EarlyStopping**  
- ✅ Live training visualization via **TensorBoard**

---

## 📂 Folder Structure

``` graphql
📦 Customer-Churn-Predictor-ANN-Streamlit
│
├── Churn_Modelling.csv                  # Raw dataset
├── ann_model.h5                         # Final trained ANN model
├── app.py                               # Streamlit application script
├── scaler.pkl                           # Saved StandardScaler
├── label_encoder_gender.pkl             # Saved LabelEncoder for Gender
├── onehot_encoder_geo.pkl               # Saved OneHotEncoder for Geography
├── prediction.ipynb                     # Notebook for loading and testing predictions
├── experiments.ipynb                    # Notebook for training, evaluation, and plots
├── requirements.txt                     # Python dependencies
├── README.md                            # Project overview and documentation
├── .gitignore                           # Ignore unnecessary files
│
└── logs/                                # TensorBoard logs
    └── fit/                             # Auto-generated training logs
```

---

## 📊 Problem Statement

Customer churn significantly impacts revenue in subscription-based services and financial institutions. This project builds a machine learning solution to predict whether a customer will exit (**churn**) based on demographic, account, and behavioral features.

---

## ⚙️ Tech Stack

| Component        | Technology          |
|------------------|---------------------|
| Language         | Python 3.11         |
| ML Framework     | TensorFlow / Keras  |
| Data Handling    | Pandas, NumPy       |
| Preprocessing    | scikit-learn        |
| Visualization    | Matplotlib, Seaborn |
| Logging          | TensorBoard         |
| Model Format     | `.h5`, `.pkl`       |

---

## 🔎 Model Architecture

The model is a fully connected **feed-forward ANN** with:

- **Input**: 12 features (after encoding)  
- **Hidden Layer 1**: 64 neurons (ReLU)  
- **Hidden Layer 2**: 32 neurons (ReLU)  
- **Output Layer**: 1 neuron (Sigmoid activation for binary classification)  

**Loss**: Binary Crossentropy  
**Optimizer**: Adam (`lr = 0.01`)  
**EarlyStopping**: Enabled (patience = 10)

---

## 📈 Evaluation Metrics

| Metric             | Value  |
|--------------------|--------|
| Accuracy           | 0.86   |
| Precision (Churn)  | 0.71   |
| Recall (Churn)     | 0.45   |
| F1-score (Churn)   | 0.55   |
| ROC-AUC Score      | 0.8566 |

> Despite class imbalance, the model achieves a strong **ROC-AUC**, indicating good overall classification performance.

---

## 🧪 How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/Mehul-kh2005/Customer-Churn-Predictor-ANN-Streamlit.git

cd Customer-Churn-Predictor-ANN-Streamlit
```

### 2. Create Virtual Environment & Install Dependencies
```bash
# 1. Create the virtual environment named `churn_pred_ANN`
conda create -n churn_pred_ANN python=3.11 -y

# 2. Activate the environment (on Linux/Mac)
conda activate churn_pred_ANN

# Install dependencies
pip install -r requirements.txt
```

### 3. Run the Notebook
Open `experiments.ipynb` in **Jupyter** or **VS Code** and run all cells.

### 4. View Training Logs with TensorBoard
```bash
tensorboard --logdir=logs/fit
```

---

## 📦 Model Artifacts

- ✅ `ann_model.h5` – Final trained Keras model  
- ✅ `scaler.pkl` – StandardScaler used for normalization  
- ✅ `label_encoder_gender.pkl` – Encodes Male/Female to 1/0  
- ✅ `onehot_encoder_geo.pkl` – Encodes Geography into dummy variables  

---

## 📚 Dataset

📄 `Churn_Modelling.csv`  
📌 **Source**: Public Kaggle banking dataset  

---

## 👨‍💻 Author

### Mehul Khandelwal - 🔗 [GitHub](https://github.com/Mehul-kh2005) | [LinkedIn](https://www.linkedin.com/in/mehulkhandelwal2005/)

---

## 💡 Inspiration

This project demonstrates how to **build**, **train**, **evaluate**, and **save** a deep learning model for real-world **binary classification** problems. While focused on **churn prediction**, the same structure can be applied to other domains like **fraud detection**, **employee attrition**, etc.

---

## ✅ TODOs (Optional Enhancements)
  
- [ ] Save and serve model with **Flask** or **FastAPI**  
- [ ] Integrate **MLflow** for experiment tracking  
- [ ] Add **hyperparameter tuning**  
- [ ] Explore **SHAP/LIME** for explainability  

---

## 📜 License

This project is open-source under the **MIT License**.
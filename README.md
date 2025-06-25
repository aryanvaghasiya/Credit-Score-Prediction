# 💳 Credit Score Prediction

A Machine Learning project for predicting an individual's credit score category (Good, Standard, Poor) based on behavioral and financial data.

---

## 📘 Project Overview

This project aims to develop and evaluate multiple ML models to predict credit scores using a real-world dataset with 27+ financial features. The goal is to build an accurate and generalizable system to assist financial institutions in assessing creditworthiness.

---

## 📂 Dataset

- **train.csv** – Contains labeled data with the `Credit Score` column.
- **test.csv** – Unlabeled dataset for final predictions.
- **Target Variable**: `Credit Score` (Good, Standard, Poor)

---

## 🔧 Preprocessing Highlights

- Cleaned mistyped numeric values stored as strings
- Imputed missing values with medians
- Handled outliers using IQR method
- Transformed complex strings (e.g., 'Credit History Age') into usable numeric features
- One-hot and label encoding applied to categorical features

---

## 🧠 Models Trained

| Model               | Accuracy |
|--------------------|----------|
| Decision Tree       | 69.85%   |
| Random Forest       | 79.22%   |
| Naive Bayes         | 57.96%   |
| XGBoost             | **79.56%** |
| AdaBoost            | 65.61%   |
| KNN                 | ~65%     |

🏆 **Final model:** XGBoost for best accuracy and balance across metrics.

---

## 🚀 Usage

```bash
# Clone the repository
git clone https://github.com/aryanvaghasiya/Credit-Score-Prediction.git
cd Credit-Score-Prediction

# Install dependencies
pip install -r requirements.txt

# Run training
python train.py

# Predict on test set
python predict.py
```

---

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score (weighted)

---

## 🧪 Dependencies

See `requirements.txt` for a full list.

---

## 👥 Contributors

- Aryan Vaghasiya (IMT2022046) — [aryan.vaghasiya@iiitb.ac.in](mailto:aryan.vaghasiya@iiitb.ac.in)
- Areen Vaghasiya (IMT2022048) — [areen.vaghasiya@iiitb.ac.in](mailto:areen.vaghasiya@iiitb.ac.in)
- Shreyank Gopalkrishna Bhat (IMT2022516) — [shreyank.bhat@iiitb.ac.in](mailto:shreyank.bhat@iiitb.ac.in)

---

## 📌 License

This project is for academic use only.

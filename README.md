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

## 📊 Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1 Score (weighted)

---

## 🧪 Dependencies

See `requirements.txt` for a full list.

-----

## How to Run

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/aryanvaghasiya/Credit-Score-Prediction.git
    cd Credit-Score-Prediction
    ```

2.  **Add Data Files:**
    Place `train.csv` and `test.csv` directly into the project's root directory.

3.  **Install Dependencies:**
    It's recommended to use a virtual environment:

    ```bash
    python -m venv venv
    # Windows: .\venv\Scripts\activate
    # macOS/Linux: source venv/bin/activate

    pip install -r requirements.txt
    ```

4.  **Run the Script:**

    ```bash
    python3 finalcreditbank.py
    ```

    This script will train models, evaluate them, and generate prediction CSV files (e.g., `xgb_submission.csv`) in the project directory.

-----
---

## 📌 License

This project is for academic use only.

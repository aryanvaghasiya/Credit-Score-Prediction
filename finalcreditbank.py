import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
from scipy import stats


def load_data(train_path="train.csv", test_path="test.csv"):
    """
    Loads training and testing data from CSV files.

    Args:
        train_path (str): Path to the training CSV file.
        test_path (str): Path to the testing CSV file.

    Returns:
        tuple: A tuple containing the train and test DataFrames.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"Train data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    return train_df, test_df

def preprocess_data(df):
    """
    Applies various preprocessing steps to the input DataFrame.

    Args:
        df (pd.DataFrame): The input DataFrame to preprocess.

    Returns:
        pd.DataFrame: The preprocessed DataFrame.
    """
    # Correcting numbers of format "_ _num_ _"
    Cols_with_underscores = ['Age','Income_Annual','Total_Current_Loans','Total_Delayed_Payments','Credit_Limit','Credit_Mix','Current_Debt_Outstanding','Monthly_Investment','Monthly_Balance']
    for col in Cols_with_underscores:
        # Check if the column exists before attempting to replace
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace('_','')

    # Convert to numeric, coercing errors
    numeric_columns = ['Age','Income_Annual','Total_Current_Loans','Total_Delayed_Payments','Credit_Limit','Current_Debt_Outstanding','Monthly_Investment','Monthly_Balance']
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Fill NaN values with median
    numeric_columns2 = ['Income_Annual','Total_Delayed_Payments','Monthly_Investment','Monthly_Balance','Total_Credit_Enquiries','Credit_Limit','Base_Salary_PerMonth']
    for col in numeric_columns2:
        if col in df.columns:
            df[col].fillna(df[col].median(), inplace=True)

    # Correcting -100 in Total_Current_Loans
    if 'Total_Current_Loans' in df.columns:
        df['Total_Current_Loans'] = df['Total_Current_Loans'].replace(-100, 0)

    # Outlier removal using IQR method
    class outlierremoval:
        def __init__(self, col):
            q1 = col.quantile(0.25)
            q3 = col.quantile(0.75)
            inter_quartile_range=q3-q1
            self.upper_whisker=q3+inter_quartile_range*1.5
            self.lower_whisker=q1-inter_quartile_range*1.5

        def remove(self, row):
            if pd.isna(row): # Handle NaN values
                return row
            if(row<=self.upper_whisker and row>=self.lower_whisker):
                return row
            elif row < self.lower_whisker:
                return self.lower_whisker
            else:
                return self.upper_whisker

    for col in ['Age', 'Base_Salary_PerMonth', 'Total_Bank_Accounts', 'Total_Credit_Cards',
                'Credit_Limit', 'Monthly_Balance', 'Income_Annual', 'Current_Debt_Outstanding',
                'Per_Month_EMI', 'Monthly_Investment']:
        if col in df.columns:
            remover = outlierremoval(df[col])
            df[col] = df[col].apply(remover.remove)

    # Correcting 'Credit_Mix', 'Profession', 'Payment_Behaviour'
    if 'Credit_Mix' in df.columns:
        df['Credit_Mix'] = df['Credit_Mix'].replace('', 'unspecified')
    if 'Profession' in df.columns:
        df['Profession'] = df['Profession'].replace('_______', 'unspecified')
    if 'Payment_Behaviour' in df.columns:
        df['Payment_Behaviour'] = df['Payment_Behaviour'].replace('!@9#%8', 'Not Specified')

    # Drop 'Name' and 'Customer_ID'
    df = df.drop(columns=['Name', 'Customer_ID'], errors='ignore')

    # Calculate total months for 'Credit_History_Age'
    def calculate_total_months(history):
        if pd.isna(history):
            return np.nan
        histlist = str(history).split()
        years = int(histlist[0]) if histlist[0].isdigit() else 0
        months = int(histlist[3]) if len(histlist) > 3 and histlist[3].isdigit() else 0
        return years * 12 + months

    if 'Credit_History_Age' in df.columns:
        df['Credit_History_Age'] = df['Credit_History_Age'].apply(calculate_total_months)
        df["Credit_History_Age"].fillna(df["Credit_History_Age"].median(), inplace=True)

    # Loan Type Encoding
    if 'Loan_Type' in df.columns:
        loan_types = set()
        for loans in df['Loan_Type'].dropna():
            for loan in loans.replace("and", "").split(','):
                loan_types.add(loan.strip())
        loan_types = [loan for loan in loan_types if loan]

        for loan_type in loan_types:
            df[f"Loan_Type_{loan_type}"] = 0

        if df['Loan_Type'].isna().sum() > 0:
            df['Loan_Type_Not Specified'] = 0

        for idx, row in df.iterrows():
            if pd.isna(row['Loan_Type']):
                df.at[idx, 'Loan_Type_Not Specified'] = 1
            else:
                for loan in str(row['Loan_Type']).replace("and", "").split(','):
                    loan = loan.strip()
                    col_name = f"Loan_Type_{loan}"
                    if col_name in df.columns: # Check if column exists (might be different between train/test unique types)
                        df.at[idx, col_name] += 1
        df = df.drop(columns=['Loan_Type'])

    # Drop 'Number' column as per notebook's implicit dropping by not using it
    df = df.drop(columns=['Number'], errors='ignore')

    return df

def encode_categorical_features(train_df, test_df):
    """
    Applies Label Encoding to specified categorical columns in both train and test DataFrames.
    Uses encoders fitted on the training data to transform the test data.

    Args:
        train_df (pd.DataFrame): Training DataFrame.
        test_df (pd.DataFrame): Testing DataFrame.

    Returns:
        tuple: A tuple containing the encoded train_df, encoded test_df, and the fitted LabelEncoders.
    """
    # Initialize LabelEncoders for each column
    month_le = LabelEncoder()
    profession_le = LabelEncoder()
    credit_mix_le = LabelEncoder()
    payment_min_amt_le = LabelEncoder()
    payment_behaviour_le = LabelEncoder()

    categorical_cols = {
        'Month': month_le,
        'Profession': profession_le,
        'Credit_Mix': credit_mix_le,
        'Payment_of_Min_Amount': payment_min_amt_le,
        'Payment_Behaviour': payment_behaviour_le
    }

    for col, encoder in categorical_cols.items():
        if col in train_df.columns:
            train_df[col] = encoder.fit_transform(train_df[col])
        if col in test_df.columns:
            # Handle unseen labels in test data
            # Combine train and test data for fitting to ensure all possible labels are learned
            # Or handle errors during transform:
            # test_df[col] = test_df[col].apply(lambda x: x if x in encoder.classes_ else encoder.classes_[0]) # A simple way
            # For simplicity and consistency with general practice, fit on combined data if possible, or handle with try-except/mapping
            # Here, we'll transform directly, assuming test data doesn't have entirely new categories not seen in training.
            # If new categories are present and `transform` throws an error, you'd need a more robust strategy.
            # A common approach is to add a placeholder for unseen labels or use OneHotEncoding (which is handled separately).
            try:
                test_df[col] = encoder.transform(test_df[col])
            except ValueError as e:
                print(f"Warning: Unseen label in column '{col}' in test set. Error: {e}")
                # A robust way to handle unseen labels is to re-map them to a specific value or the most frequent one
                # For this example, we'll just let the error happen or ensure data consistency before calling this.
                # A safer approach for production would be:
                test_df[col] = test_df[col].map(lambda s: f'<unseen_label_{s}>' if s not in encoder.classes_ else s)
                encoder.fit(pd.concat([train_df[col], test_df[col].astype(str).loc[test_df[col].astype(str).str.startswith('<unseen_label_>') == False]]))
                test_df[col] = test_df[col].astype(str).apply(lambda s: encoder.transform([s])[0] if s in encoder.classes_ else encoder.transform(['<unseen_label_mapping_value>'])[0])

    return train_df, test_df, month_le, profession_le, credit_mix_le, payment_min_amt_le, payment_behaviour_le

def print_metrics(model_name, y_true_encoded, y_pred_encoded, target_names=None):
    """
    Prints classification performance metrics.

    Args:
        model_name (str): Name of the model.
        y_true_encoded (np.array): True labels (encoded).
        y_pred_encoded (np.array): Predicted labels (encoded).
        target_names (list, optional): List of target class names. Defaults to None.
    """
    accuracy = accuracy_score(y_true_encoded, y_pred_encoded)
    precision = precision_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
    recall = recall_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)
    f1 = f1_score(y_true_encoded, y_pred_encoded, average='weighted', zero_division=0)

    print(f"\n--- {model_name} ---")
    print(f"Test Accuracy: {accuracy:.4f}")
    print(f"Test Precision: {precision:.4f}")
    print(f"Test Recall: {recall:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    if target_names:
        print("Classification Report:\n", classification_report(y_true_encoded, y_pred_encoded, target_names=target_names, zero_division=0))


def train_and_evaluate_xgboost(X_train, y_train_encoded, X_test, y_test_encoded, label_encoder, test_data_original):
    """
    Trains and evaluates an XGBoost Classifier and saves predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train_encoded (np.array): Encoded training labels.
        X_test (pd.DataFrame): Test features.
        y_test_encoded (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_original (pd.DataFrame): Original test data (for ID column).
    """
    param_distributions_xgb = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'max_depth': [3, 5, 7, 10, 15],
        'min_child_weight': [1, 5, 10],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.3],
        'reg_alpha': [0, 0.01, 0.1],
        'reg_lambda': [1, 1.5, 2]
    }

    xgboost_clf = xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    random_search_xgb = RandomizedSearchCV(estimator=xgboost_clf, param_distributions=param_distributions_xgb,
                                           n_iter=50, cv=5, n_jobs=-1, verbose=0, scoring='accuracy', random_state=42)

    random_search_xgb.fit(X_train, y_train_encoded)
    best_xgb = random_search_xgb.best_estimator_

    y_pred_train = best_xgb.predict(X_train)
    y_pred_test = best_xgb.predict(X_test)

    print("\n--- XGBoost Classifier ---")
    print("Best Parameters:", random_search_xgb.best_params_)
    print(f"Training Accuracy: {accuracy_score(y_train_encoded, y_pred_train):.4f}")
    print_metrics("XGBoost", y_test_encoded, y_pred_test, target_names=label_encoder.classes_)

    # Ensure final test data has the same columns as X_train
    x_final_test = test_data_original.drop(columns=['ID'], errors='ignore').reindex(columns=X_train.columns, fill_value=0)
    y_pred_final_test_encoded = best_xgb.predict(x_final_test)
    y_pred_final_test = label_encoder.inverse_transform(y_pred_final_test_encoded)

    output_xgb = pd.DataFrame({'ID': test_data_original['ID'], 'Credit_Score': y_pred_final_test})
    output_xgb.to_csv('xgb_submission.csv', index=False)
    print("XGBoost predictions saved to xgb_submission.csv")


def train_and_evaluate_gaussiannb(X_train, y_train_encoded, X_test, y_test_encoded, label_encoder, test_data_encoded_original):
    """
    Trains and evaluates a Gaussian Naive Bayes Classifier and saves predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train_encoded (np.array): Encoded training labels.
        X_test (pd.DataFrame): Test features.
        y_test_encoded (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_encoded_original (pd.DataFrame): Original test data (for ID column, after preprocessing and encoding).
    """
    naive_bayes_clf = GaussianNB()
    naive_bayes_clf.fit(X_train, y_train_encoded)

    y_pred_train = naive_bayes_clf.predict(X_train)
    y_pred_test = naive_bayes_clf.predict(X_test)

    print("\n--- Gaussian Naive Bayes Classifier ---")
    print(f"Training Accuracy: {accuracy_score(y_train_encoded, y_pred_train):.4f}")
    print_metrics("Gaussian Naive Bayes", y_test_encoded, y_pred_test, target_names=label_encoder.classes_)

    # Ensure final test data has the same columns as X_train
    # Note: test_data_encoded_original might already be aligned, reindex to be safe.
    x_final_test = test_data_encoded_original.drop(columns=['ID','Credit_Score'], errors='ignore').reindex(columns=X_train.columns, fill_value=0)
    y_pred_final_test_encoded = naive_bayes_clf.predict(x_final_test)
    y_pred_final_test = label_encoder.inverse_transform(y_pred_final_test_encoded)

    output_naive_bayes = pd.DataFrame({'ID': test_data_encoded_original['ID'], 'Credit_Score': y_pred_final_test})
    output_naive_bayes.to_csv('naive_bayes_submission.csv', index=False)
    print("Naive Bayes predictions saved to naive_bayes_submission.csv")

def train_and_evaluate_decisiontree(X_train, y_train_encoded, X_test, y_test_encoded, label_encoder, test_data_original):
    """
    Trains and evaluates a Decision Tree Classifier and saves predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train_encoded (np.array): Encoded training labels.
        X_test (pd.DataFrame): Test features.
        y_test_encoded (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_original (pd.DataFrame): Original test data (for ID column).
    """
    param_distributions_dt = {
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    decisiontree_clf = DecisionTreeClassifier(random_state=42)
    random_search_dt = RandomizedSearchCV(estimator=decisiontree_clf, param_distributions=param_distributions_dt,
                                          n_iter=20, cv=5, n_jobs=-1, verbose=0, scoring='accuracy', random_state=42)
    random_search_dt.fit(X_train, y_train_encoded)
    best_dt = random_search_dt.best_estimator_
    y_pred_dt_test = best_dt.predict(X_test)

    print("\n--- Decision Tree Classifier ---")
    print("Best Parameters:", random_search_dt.best_params_)
    print_metrics("Decision Tree", y_test_encoded, y_pred_dt_test, target_names=label_encoder.classes_)

    x_final_test = test_data_original.drop(columns=['ID'], errors='ignore').reindex(columns=X_train.columns, fill_value=0)
    y_pred_final_test_encoded_dt = best_dt.predict(x_final_test)
    y_pred_final_test_dt = label_encoder.inverse_transform(y_pred_final_test_encoded_dt)
    output_dt = pd.DataFrame({'ID': test_data_original['ID'], 'Credit_Score': y_pred_final_test_dt})
    output_dt.to_csv('decisiontree_submission.csv', index=False)
    print("Decision Tree predictions saved to decisiontree_submission.csv")

def train_and_evaluate_randomforest(X_train, y_train_encoded, X_test, y_test_encoded, label_encoder, test_data_original):
    """
    Trains and evaluates a Random Forest Classifier and saves predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train_encoded (np.array): Encoded training labels.
        X_test (pd.DataFrame): Test features.
        y_test_encoded (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_original (pd.DataFrame): Original test data (for ID column).
    """
    param_distributions_rf = {
        'n_estimators': [100, 150, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }
    randomforest_clf = RandomForestClassifier(random_state=42)
    random_search_rf = RandomizedSearchCV(estimator=randomforest_clf, param_distributions=param_distributions_rf,
                                          n_iter=20, cv=5, n_jobs=-1, verbose=0, scoring='accuracy', random_state=42)
    random_search_rf.fit(X_train, y_train_encoded)
    best_rf = random_search_rf.best_estimator_
    y_pred_rf_test = best_rf.predict(X_test)

    print("\n--- Random Forest Classifier ---")
    print("Best Parameters:", random_search_rf.best_params_)
    print_metrics("Random Forest", y_test_encoded, y_pred_rf_test, target_names=label_encoder.classes_)

    x_final_test = test_data_original.drop(columns=['ID'], errors='ignore').reindex(columns=X_train.columns, fill_value=0)
    y_pred_final_test_encoded_rf = best_rf.predict(x_final_test)
    y_pred_final_test_rf = label_encoder.inverse_transform(y_pred_final_test_encoded_rf)
    output_rf = pd.DataFrame({'ID': test_data_original['ID'], 'Credit_Score': y_pred_final_test_rf})
    output_rf.to_csv('randomforest_submission.csv', index=False)
    print("Random Forest predictions saved to randomforest_submission.csv")

def train_and_evaluate_adaboost(X_train, y_train_encoded, X_test, y_test_encoded, label_encoder, test_data_original):
    """
    Trains and evaluates an AdaBoost Classifier and saves predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train_encoded (np.array): Encoded training labels.
        X_test (pd.DataFrame): Test features.
        y_test_encoded (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_original (pd.DataFrame): Original test data (for ID column).
    """
    param_distributions_ada = {
        'n_estimators': [50, 100, 150, 200],
        'learning_rate': [0.01, 0.1, 0.5, 1]
    }
    adaboost_clf = AdaBoostClassifier(random_state=42)
    random_search_ada = RandomizedSearchCV(estimator=adaboost_clf, param_distributions=param_distributions_ada,
                                           n_iter=20, cv=5, n_jobs=-1, verbose=0, scoring='accuracy', random_state=42)
    random_search_ada.fit(X_train, y_train_encoded)
    best_ada = random_search_ada.best_estimator_
    y_pred_ada_test = best_ada.predict(X_test)

    print("\n--- AdaBoost Classifier ---")
    print("Best Parameters:", random_search_ada.best_params_)
    print_metrics("AdaBoost", y_test_encoded, y_pred_ada_test, target_names=label_encoder.classes_)

    x_final_test = test_data_original.drop(columns=['ID'], errors='ignore').reindex(columns=X_train.columns, fill_value=0)
    y_pred_final_test_encoded_ada = best_ada.predict(x_final_test)
    y_pred_final_test_ada = label_encoder.inverse_transform(y_pred_final_test_encoded_ada)
    output_ada = pd.DataFrame({'ID': test_data_original['ID'], 'Credit_Score': y_pred_final_test_ada})
    output_ada.to_csv('adaboost_submission.csv', index=False)
    print("AdaBoost predictions saved to adaboost_submission.csv")

def train_and_evaluate_knn(X_train_knn, y_train_knn, X_test_knn, y_test_knn, label_encoder, test_data_original):
    """
    Trains and evaluates a K-Nearest Neighbors Classifier.
    Note: This function does not generate a submission file in this version.

    Args:
        X_train_knn (pd.DataFrame): Training features.
        y_train_knn (np.array): Encoded training labels.
        X_test_knn (pd.DataFrame): Test features.
        y_test_knn (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_original (pd.DataFrame): Original test data (for ID column).
    """
    K = []
    training = []
    test = []
    scores = {}

    for k in range(2, 21):
        clf = KNeighborsClassifier(n_neighbors = k)
        clf.fit(X_train_knn, y_train_knn)

        training_score = clf.score(X_train_knn, y_train_knn)
        test_score = clf.score(X_test_knn, y_test_knn)
        K.append(k)
        training.append(training_score)
        test.append(test_score)
        scores[k] = [training_score, test_score]

    print("\n--- K-Nearest Neighbors Classifier ---")
    print("K-NN Scores:", scores)

    # To generate a submission for KNN, you would typically:
    # 1. Select the best 'k' based on validation/test scores.
    # 2. Train a KNN model with the best 'k' on the full training data (or X_train).
    # 3. Predict on the `test_data_original` (after preprocessing and column alignment).
    # 4. Save the predictions to a CSV.
    # For now, keeping it consistent with your original notebook's KNN section.


def train_and_evaluate_svm(X_train, y_train_encoded, X_test, y_test_encoded, label_encoder, test_data_original):
    """
    Trains and evaluates a Support Vector Machine Classifier and saves predictions.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train_encoded (np.array): Encoded training labels.
        X_test (pd.DataFrame): Test features.
        y_test_encoded (np.array): Encoded test labels.
        label_encoder (LabelEncoder): Fitted LabelEncoder for target variable.
        test_data_original (pd.DataFrame): Original test data (for ID column).
    """
    # SVM can be very computationally expensive on large datasets.
    # It is highly recommended to scale data for SVM.
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # You can experiment with different kernels: 'linear', 'poly', 'sigmoid', 'rbf'
    # Add C parameter for regularization (default is 1.0)
    svm_model = SVC(kernel='rbf', random_state=42, C=1.0)
    svm_model.fit(X_train_scaled, y_train_encoded) # Fit with scaled data

    y_pred_train = svm_model.predict(X_train_scaled)
    y_pred_test = svm_model.predict(X_test_scaled)

    print("\n--- Support Vector Machine Classifier ---")
    print(f"Training Accuracy: {accuracy_score(y_train_encoded, y_pred_train):.4f}")
    print_metrics("Support Vector Machine", y_test_encoded, y_pred_test, target_names=label_encoder.classes_)

    # Predict on the actual test data for submission
    # Ensure final test data has the same columns as X_train and scale it
    x_final_test_scaled = scaler.transform(test_data_original.drop(columns=['ID'], errors='ignore').reindex(columns=X_train.columns, fill_value=0))
    y_pred_final_test_encoded_svm = svm_model.predict(x_final_test_scaled)
    y_pred_final_test_svm = label_encoder.inverse_transform(y_pred_final_test_encoded_svm)

    output_svm = pd.DataFrame({
        'ID': test_data_original['ID'],
        'Credit_Score': y_pred_final_test_svm
    })
    output_svm.to_csv('svm_submission.csv', index=False)
    print("SVM predictions saved to svm_submission.csv")


if __name__ == "__main__":
    # Load raw data
    train_data_raw, test_data_raw = load_data()

    # Make copies for different encoding strategies if needed
    # (Though for this specific notebook, Label Encoding and One-Hot Encoding are applied sequentially
    # based on model choice, so deep copies ensure original state for each branch)
    train_data_for_label_encoding = train_data_raw.copy()
    test_data_for_label_encoding = test_data_raw.copy()
    train_data_for_one_hot = train_data_raw.copy()
    test_data_for_one_hot = test_data_raw.copy()

    # --- Preprocessing and Model Training for Label Encoding based models (XGBoost, RF, DT, AdaBoost, KNN, SVM) ---
    print("\n--- Processing for Label Encoded Models ---")
    train_data_processed_le = preprocess_data(train_data_for_label_encoding)
    test_data_processed_le = preprocess_data(test_data_for_label_encoding)

    # Before encoding, ensure that train and test have a consistent set of columns,
    # especially for 'Loan_Type' features which are created dynamically.
    # And preserve 'ID' and 'Credit_Score' for later use.
    train_cols_le = set(train_data_processed_le.columns)
    test_cols_le = set(test_data_processed_le.columns)

    # Get common columns plus the target and ID if they exist in train
    common_features_le = list(train_cols_le.intersection(test_cols_le).difference(['ID', 'Credit_Score']))
    train_data_processed_le = train_data_processed_le[common_features_le + ['ID', 'Credit_Score']].copy()
    test_data_processed_le = test_data_processed_le[common_features_le + ['ID']].copy()

    # Align columns explicitly for test set (fill_value=0 for new Loan_Type_X columns that might appear only in train)
    train_data_processed_le, test_data_processed_le = train_data_processed_le.align(
        test_data_processed_le, join='outer', axis=1, fill_value=0)

    # Drop ID and Credit_Score from alignment if they were included
    train_data_processed_le.drop(columns=['ID'], errors='ignore', inplace=True)
    test_data_processed_le.drop(columns=['ID'], errors='ignore', inplace=True)


    train_data_encoded_le, test_data_encoded_le, month_le, profession_le, credit_mix_le, payment_min_amt_le, payment_behaviour_le = \
        encode_categorical_features(train_data_processed_le.copy(), test_data_processed_le.copy())

    # Re-extract ID for submission after encoding
    train_ids_le = train_data_encoded_le['ID'] if 'ID' in train_data_encoded_le.columns else pd.Series([], dtype='int64')
    test_ids_le = test_data_encoded_le['ID'] if 'ID' in test_data_encoded_le.columns else pd.Series([], dtype='int64')

    # Drop 'ID' before splitting features and target
    train_data_encoded_le = train_data_encoded_le.drop(columns=['ID'], errors='ignore')
    test_data_encoded_le = test_data_encoded_le.drop(columns=['ID'], errors='ignore')


    # Split data for training and testing
    # Stratify by Credit_Score to maintain class distribution in splits
    X_train_le = train_data_encoded_le.drop(columns=['Credit_Score'])
    y_train_le = train_data_encoded_le['Credit_Score']

    # Splitting training data into train and validation sets
    X_train_split_le, X_test_split_le, y_train_split_le, y_test_split_le = train_test_split(
        X_train_le, y_train_le, test_size=0.2, random_state=42, stratify=y_train_le
    )

    label_encoder = LabelEncoder()
    y_train_encoded_le = label_encoder.fit_transform(y_train_split_le)
    y_test_encoded_le = label_encoder.transform(y_test_split_le)

    # Ensure test_data_encoded_le used for final predictions has the same columns as X_train_split_le
    # and preserve the original 'ID' for submission.
    # The `test_data_raw` already has the 'ID' column, use it directly for submission dataframes.
    final_test_data_for_le_models = test_data_raw.copy()
    final_test_data_for_le_models = preprocess_data(final_test_data_for_le_models)
    final_test_data_for_le_models, _ ,_,_,_,_,_ = encode_categorical_features(final_test_data_for_le_models, final_test_data_for_le_models.copy()) # Encode test data itself
    final_test_data_for_le_models = final_test_data_for_le_models.drop(columns=['ID'], errors='ignore').reindex(columns=X_train_split_le.columns, fill_value=0)
    final_test_data_for_le_models['ID'] = test_data_raw['ID'] # Add ID back for submission

    # Run models with Label Encoding
    train_and_evaluate_xgboost(X_train_split_le, y_train_encoded_le, X_test_split_le, y_test_encoded_le, label_encoder, final_test_data_for_le_models)
    train_and_evaluate_decisiontree(X_train_split_le, y_train_encoded_le, X_test_split_le, y_test_encoded_le, label_encoder, final_test_data_for_le_models)
    train_and_evaluate_randomforest(X_train_split_le, y_train_encoded_le, X_test_split_le, y_test_encoded_le, label_encoder, final_test_data_for_le_models)
    train_and_evaluate_adaboost(X_train_split_le, y_train_encoded_le, X_test_split_le, y_test_encoded_le, label_encoder, final_test_data_for_le_models)
    train_and_evaluate_knn(X_train_split_le, y_train_encoded_le, X_test_split_le, y_test_encoded_le, label_encoder, final_test_data_for_le_models)
    train_and_evaluate_svm(X_train_split_le, y_train_encoded_le, X_test_split_le, y_test_encoded_le, label_encoder, final_test_data_for_le_models)


    # --- Preprocessing and Model Training for One-Hot Encoding based models (Gaussian Naive Bayes) ---
    print("\n--- Processing for One-Hot Encoded Models ---")
    train_data_processed_ohe = preprocess_data(train_data_for_one_hot)
    test_data_processed_ohe = preprocess_data(test_data_for_one_hot)

    # Define categorical columns to be One-Hot Encoded
    categorical_columns_ohe = ['Month', 'Profession', 'Credit_Mix', 'Payment_of_Min_Amount', 'Payment_Behaviour']

    # Perform One-Hot Encoding
    train_data_encoded_ohe = pd.get_dummies(train_data_processed_ohe, columns=categorical_columns_ohe, drop_first=True)
    test_data_encoded_ohe = pd.get_dummies(test_data_processed_ohe, columns=categorical_columns_ohe, drop_first=True)

    # Align columns - crucial for consistency between train and test after OHE
    # This adds columns present in train but not test (and vice-versa) and fills with 0
    train_cols_ohe = set(train_data_encoded_ohe.columns)
    test_cols_ohe = set(test_data_encoded_ohe.columns)

    common_features_ohe = list(train_cols_ohe.intersection(test_cols_ohe).difference(['ID', 'Credit_Score']))

    # Ensure target and ID are handled separately
    X_train_ohe_full = train_data_encoded_ohe[common_features_ohe + ['ID']].copy()
    y_train_ohe_full = train_data_encoded_ohe['Credit_Score'].copy()
    X_test_ohe_full = test_data_encoded_ohe[common_features_ohe + ['ID']].copy()

    # Add columns that are in train_data_encoded_ohe but not in test_data_encoded_ohe, fill with 0
    missing_in_test_ohe = list(train_cols_ohe - test_cols_ohe)
    for col in missing_in_test_ohe:
        if col != 'Credit_Score' and col != 'ID':
            X_test_ohe_full[col] = 0

    # Add columns that are in test_data_encoded_ohe but not in train_data_encoded_ohe, fill with 0
    missing_in_train_ohe = list(test_cols_ohe - train_cols_ohe)
    for col in missing_in_train_ohe:
        if col != 'Credit_Score' and col != 'ID':
            X_train_ohe_full[col] = 0

    # Ensure consistent order of columns
    X_test_ohe_full = X_test_ohe_full.reindex(columns=X_train_ohe_full.columns, fill_value=0)


    # Drop 'ID' before splitting features and target for training
    X_train_ohe = X_train_ohe_full.drop(columns=['ID'], errors='ignore')
    X_test_ohe_for_split = X_test_ohe_full.drop(columns=['ID'], errors='ignore') # For train/test split of training data

    # Split training data into train and validation sets for OHE models
    X_train_split_ohe, X_test_split_ohe, y_train_split_ohe, y_test_split_ohe = train_test_split(
        X_train_ohe, y_train_ohe_full, test_size=0.2, random_state=42, stratify=y_train_ohe_full
    )

    y_train_encoded_ohe = label_encoder.fit_transform(y_train_split_ohe)
    y_test_encoded_ohe = label_encoder.transform(y_test_split_ohe)

    # Prepare the final test data for prediction by Naive Bayes
    final_test_data_for_ohe_models = test_data_raw.copy()
    final_test_data_for_ohe_models = preprocess_data(final_test_data_for_ohe_models)
    final_test_data_for_ohe_models = pd.get_dummies(final_test_data_for_ohe_models, columns=categorical_columns_ohe, drop_first=True)
    # Align columns of final test data with the training features
    final_test_data_for_ohe_models_aligned = final_test_data_for_ohe_models.drop(columns=['ID'], errors='ignore').reindex(columns=X_train_split_ohe.columns, fill_value=0)
    final_test_data_for_ohe_models_aligned['ID'] = test_data_raw['ID'] # Add ID back for submission

    # Run Naive Bayes with One-Hot Encoding
    train_and_evaluate_gaussiannb(X_train_split_ohe, y_train_encoded_ohe, X_test_split_ohe, y_test_encoded_ohe, label_encoder, final_test_data_for_ohe_models_aligned)

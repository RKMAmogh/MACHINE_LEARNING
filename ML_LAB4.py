# BL.EN.U4CSE23109 
# Mental Health Dataset Classification - Lab 4 Complete Implementation


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# For Classification tasks
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

# For Regression tasks
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Helper function for MAPE calculation
def cal_mape(y_true, y_pred):
    """
    Calculates the Mean Absolute Percentage Error (MAPE).
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero
    non_zero = y_true != 0
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# Load the Mental Health dataset
print("=== Loading Mental Health Dataset ===")
df_full = pd.read_csv('MHDS.csv')
print(f"Full dataset shape: {df_full.shape}")

# Take only first 500 rows for faster processing
df = df_full.head(500).copy()  # Add .copy() to avoid SettingWithCopyWarning
print(f"Using subset dataset shape: {df.shape}")
print(f"Features: {list(df.columns)}")
print(f"Sample data:\n{df.head()}")
print(f"Data types:\n{df.dtypes}")
print(f"Missing values:\n{df.isnull().sum()}")

# Encode categorical variables (including Stress_Level which contains 'Low', 'Medium', 'High')
le_dict = {}
cat_col = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 'Severity', 'Consultation_History', 'Stress_Level', 'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption', 'Medication_Usage']

for col in cat_col:
    if col in df.columns:
        le_dict[col] = LabelEncoder()
        df[col] = le_dict[col].fit_transform(df[col])

# Use Mental_Health_Condition as the target variable for classification
target_col = 'Mental_Health_Condition'
feature_cols = [col for col in df.columns if col not in ['User_ID', target_col]]

print(f"\nTarget variable: {target_col}")
print(f"Feature variables: {feature_cols}")
print(f"Target distribution:\n{df[target_col].value_counts()}")

# Prepare the data
X = df[feature_cols].values
y = df[target_col].values

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target vector shape: {y.shape}")
print(f"Class distribution: {np.bincount(y)}")

# Split the data for all assignments
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

print(f"\nTraining set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training class distribution: {np.bincount(y_train)}")
print(f"Test class distribution: {np.bincount(y_test)}")

print("\n" + "="*80)
print("ASSIGNMENT A1: CLASSIFICATION EVALUATION")
print("="*80)

# --- Functions for A1 ---
def cal_matrix(y_true, y_pred):
    """
    Calculates confusion matrix, precision, recall, and f1-score.
    """
    conf_matrix = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    return conf_matrix, precision, recall, f1

def train_and_evaluate_classifier(X_train, y_train, X_test, y_test, k=5):
    """
    Trains a KNN classifier and evaluates it on both training and test data.
    """
    # Initialize and train the model
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)

    # Make predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Get metrics for training data
    train_metrics = cal_matrix(y_train, y_train_pred)

    # Get metrics for test data
    test_metrics = cal_matrix(y_test, y_test_pred)

    return train_metrics, test_metrics

# Train the model and get metrics
(train_cm, train_precision, train_recall, train_f1), (test_cm, test_precision, test_recall, test_f1) = train_and_evaluate_classifier(X_train, y_train, X_test, y_test)

# Print the results
print("--- Training Data Metrics ---")
print(f"Confusion Matrix:\n{train_cm}")
print(f"Precision: {train_precision:.4f}")
print(f"Recall: {train_recall:.4f}")
print(f"F1-Score: {train_f1:.4f}")

print("\n--- Test Data Metrics ---")
print(f"Confusion Matrix:\n{test_cm}")
print(f"Precision: {test_precision:.4f}")
print(f"Recall: {test_recall:.4f}")
print(f"F1-Score: {test_f1:.4f}")

print("\n--- Analysis of Model Fit ---")
if train_f1 > 0.95 and test_f1 < 0.8:
    print("Inference: The model is likely OVERFITTING. It performs excellently on training data but poorly on test data.")
elif train_f1 < 0.7:
    print("Inference: The model is likely UNDERFITTING. It performs poorly on both training and test data.")
else:
    print("Inference: The model seems to have a REGULAR FIT. It performs well on both training and test data without a significant drop-off.")

print("\n" + "="*80)
print("ASSIGNMENT A2: REGRESSION SCORE ANALYSIS")
print("="*80)

# For A2, we'll use Sleep_Hours as a target for regression
# We'll predict Sleep_Hours using other numerical features
print("Using Sleep_Hours as target variable for regression analysis...")

# Prepare regression data - using numerical features and encoded categorical features
numerical_features = ['Age', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage', 'Stress_Level']  # Stress_Level is now encoded
available_features = [col for col in numerical_features if col in df.columns]
X_reg = df[available_features].values
y_reg = df['Sleep_Hours'].values

# Split regression data
X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)

# Train regression model
reg_model = LinearRegression()
reg_model.fit(X_reg_train, y_reg_train)
y_reg_pred = reg_model.predict(X_reg_test)

# Calculate regression metrics
def get_regression_metrics(y_true, y_pred):
    """
    Calculates MSE, RMSE, MAPE, and R2 score for a regression model.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = cal_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return mse, rmse, mape, r2

mse, rmse, mape, r2 = get_regression_metrics(y_reg_test, y_reg_pred)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"R-squared (R2) Score: {r2:.4f}")

print("\n--- Analysis of Results ---")
print("Analysis: A high R2 (close to 1) indicates a good fit. Lower MSE, RMSE, and MAPE values indicate better accuracy.")

print("\n" + "="*80)
print("ASSIGNMENT A3, A4, A5: KNN VISUALIZATION")
print("="*80)

# For visualization, we'll use 2 numerical features from the Mental Health data
# We'll select Age and Sleep_Hours for 2D visualization
X_vis = df[['Age', 'Sleep_Hours']].values
y_vis = df[target_col].values

# Split visualization data
X_vis_train, X_vis_test, y_vis_train, y_vis_test = train_test_split(X_vis, y_vis, test_size=0.3, random_state=42, stratify=y_vis)

def plot_data(X, y, title):
    """
    Creates a scatter plot of the data, colored by class.
    """
    plt.figure(figsize=(10, 8))
    unique_classes = np.unique(y)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    for i, class_val in enumerate(unique_classes):
        mask = y == class_val
        plt.scatter(X[mask, 0], X[mask, 1], c=[colors[i]], label=f'Class {class_val}', 
                   edgecolor='k', s=50, alpha=0.7)
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Sleep Hours", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_decision_boundary(X_train, y_train, k_value):
    """
    Trains a KNN, classifies a test grid, and plots the decision boundary.
    """
    # Setup classifier
    clf = KNeighborsClassifier(n_neighbors=k_value)
    clf.fit(X_train, y_train)

    # Generate test data grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.5),
                         np.arange(y_min, y_max, 0.1))
    X_test_grid = np.c_[xx.ravel(), yy.ravel()]

    # Classify the points in the grid
    Z = clf.predict(X_test_grid)
    Z = Z.reshape(xx.shape)

    # Define colors
    unique_classes = np.unique(y_train)
    colors = plt.cm.Set1(np.linspace(0, 1, len(unique_classes)))
    
    # Plot the decision boundary
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.3, levels=len(unique_classes)-1)

    # Plot the training points on top
    for i, class_val in enumerate(unique_classes):
        mask = y_train == class_val
        plt.scatter(X_train[mask, 0], X_train[mask, 1], c=[colors[i]], 
                   label=f'Class {class_val}', edgecolor='k', s=50, alpha=0.8)
    
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.title(f"KNN Decision Boundary (k={k_value}) - Mental Health Classification", fontsize=14, fontweight='bold')
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Sleep Hours", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# A3: Plot training data
print("A3: Generating and plotting training data")
plot_data(X_vis_train, y_vis_train, "A3: Scatter Plot of Mental Health Training Data (Age vs Sleep Hours)")

# A4: Plot decision boundary for k=3
print("A4: Plotting decision boundary for k=3")
plot_decision_boundary(X_vis_train, y_vis_train, k_value=3)

# A5: Plot decision boundaries for different k values
print("A5: Observing boundary changes with different k values")
k_values_to_test = [1, 5, 15]
for k in k_values_to_test:
    plot_decision_boundary(X_vis_train, y_vis_train, k_value=k)

print("\n" + "="*80)
print("ASSIGNMENT A6: APPLYING TO PROJECT DATA")
print("="*80)

# A6: Apply to the actual Mental Health project data
print("Using Mental Health data with Age and Sleep Hours for visualization")

# A6.1: Plot the project data
plot_data(X_vis, y_vis, "A6: Scatter Plot of Mental Health Project Data (Age vs Sleep Hours)")

# A6.2: Observe decision boundaries for different k values
k_values_project = [1, 3, 10, 25]
for k in k_values_project:
    plot_decision_boundary(X_vis, y_vis, k_value=k)

print("\n" + "="*80)
print("ASSIGNMENT A7: HYPER-PARAMETER TUNING WITH GRIDSEARCHCV")
print("="*80)

def find_ideal_k(X_train, y_train):
    """
    Uses GridSearchCV to find the best 'k' for a KNN classifier.
    """
    # Define the parameter grid for 'k'
    param_grid = {'n_neighbors': np.arange(1, 32, 2)}

    # Initialize the KNN classifier
    knn = KNeighborsClassifier()

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)

    # Fit the grid search to the data
    grid_search.fit(X_train, y_train)

    # Return the best k value and the best score
    best_k = grid_search.best_params_['n_neighbors']
    best_score = grid_search.best_score_

    return best_k, best_score

# Use the full Mental Health dataset for hyperparameter tuning
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Find the ideal 'k' using the training subset
ideal_k, best_cv_score = find_ideal_k(X_train_full, y_train_full)

print(f"The ideal 'k' value found using GridSearchCV is: {ideal_k}")
print(f"The cross-validation accuracy at this k is: {best_cv_score:.4f}")

# Test the best model on the test set
best_model = KNeighborsClassifier(n_neighbors=ideal_k)
best_model.fit(X_train_full, y_train_full)
y_test_pred_best = best_model.predict(X_test_full)

# Calculate final test metrics
test_cm_best, test_precision_best, test_recall_best, test_f1_best = cal_matrix(y_test_full, y_test_pred_best)

print(f"\n--- Final Test Results with Best k={ideal_k} ---")
print(f"Confusion Matrix:\n{test_cm_best}")
print(f"Precision: {test_precision_best:.4f}")
print(f"Recall: {test_recall_best:.4f}")
print(f"F1-Score: {test_f1_best:.4f}")

print("\n" + "="*80)
print("SUMMARY OF ALL ASSIGNMENTS")
print("="*80)

print("A1: Classification Evaluation - Completed with Mental Health data")
print("A2: Regression Analysis - Completed using Sleep Hours prediction")
print("A3: Data Visualization - Completed with Age vs Sleep Hours scatter plot")
print("A4: Decision Boundary (k=3) - Completed")
print("A5: Multiple k-values - Completed with k=1,5,15")
print("A6: Project Data Application - Completed with actual Mental Health data")
print("A7: Hyperparameter Tuning - Completed with GridSearchCV")

print(f"\nBest performing model: k={ideal_k} with CV accuracy: {best_cv_score:.4f}")
print(f"Final test F1-score: {test_f1_best:.4f}")

print("\n=== LAB 4 COMPLETED SUCCESSFULLY ===")
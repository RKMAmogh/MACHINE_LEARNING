# BL.EN.U4CSE23109

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Import for data splitting
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# --- A1, A2, A3: Regression Models & Metrics ---
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

# --- A4, A5, A6, A7: Clustering Models & Metrics ---
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

# === CONFIG ===
# Path to your CSV file
CSV_FILE_PATH = "MHDS.csv"
# Use a 70/30 split for training and testing data
TEST_SIZE = 0.3
# A fixed random state ensures that the data split is the same every time we run the code
RANDOM_STATE = 42


# DATA LOADING & PREPARATION FUNCTIONS


def load_and_prepare_data(csv_path, nrows=500):
    """
    Loads the CSV file and prepares the data for machine learning.
    Args:
        csv_path (str): Path to the CSV file.
        nrows (int): Number of rows to load (default: 500 for faster processing).

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The prepared feature data (X).
            - np.ndarray: The target data (y) - using Stress_Level as target.
            - pd.DataFrame: The original dataframe for reference.
    """
    # Load only the first 500 rows of the CSV file for faster processing
    df = pd.read_csv(csv_path, nrows=nrows)
    print(f"Loaded dataset with shape: {df.shape} (limited to first {nrows} rows)")
    print(f"Columns: {list(df.columns)}")
    
    # Handle missing values
    print(f"Missing values per column:\n{df.isnull().sum()}")
    df = df.dropna()  # Simple approach - drop rows with missing values
    print(f"Dataset shape after removing missing values: {df.shape}")
    
    # Prepare features and target
    # We'll use Stress_Level as our regression target
    tar_col = 'Stress_Level'
    
    # Select numerical features for initial analysis
    num_fea = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 
                         'Social_Media_Usage']
    
    # Encode categorical features (including Diet_Quality and Stress_Level which contain text)
    cat_fea = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 
                           'Severity', 'Consultation_History', 'Smoking_Habit', 
                           'Alcohol_Consumption', 'Medication_Usage', 'Diet_Quality', 'Stress_Level']
    
    # Create a copy for processing
    df_processed = df.copy()
    
    # Label encode categorical variables
    lab_encod = {}
    for col in cat_fea:
        if col in df_processed.columns:
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
            lab_encod[col] = le
            print(f"Encoded {col}: {len(le.classes_)} unique values")
    
    # Ensure numerical columns are actually numeric
    for col in num_fea:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Check for any remaining non-numeric data
    print(f"Data types after encoding:")
    for col in df_processed.columns:
        if col not in ['User_ID']:
            print(f"  {col}: {df_processed[col].dtype}")
    
    # Drop any rows with NaN values that might have been created during conversion
    df_processed = df_processed.dropna()
    print(f"Final dataset shape after cleaning: {df_processed.shape}")
    
    # Prepare feature matrix (X) - all columns except User_ID and target
    fea_col = [col for col in df_processed.columns if col not in ['User_ID', tar_col]]
    X = df_processed[fea_col].values
    
    # Prepare target vector (y) - Stress_Level is now encoded as numbers
    y = df_processed[tar_col].values
    
    # Convert target to numeric if it's not already (safety check)
    if y.dtype == 'object':
        le_target = LabelEncoder()
        y = le_target.fit_transform(y)
        print(f"Target variable encoded: {le_target.classes_}")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print(f"Final feature matrix shape: {X_scaled.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Target values range: {y.min()} to {y.max()}")
    print(f"Feature columns: {fea_col}")
    
    return X_scaled, y, df, fea_col, scaler


# REGRESSION FUNCTIONS (A1, A2, A3)


def train_linear_regression(X_train, y_train):
    """
    Trains a linear regression model.
    
    Args:
        X_train (np.ndarray): Training feature data.
        y_train (np.ndarray): Training target data.
        
    Returns:
        LinearRegression: The fitted linear regression model object.
    """
    reg_model = LinearRegression()
    reg_model.fit(X_train, y_train)
    return reg_model

def get_regression_predictions(model, X):
    """
    Makes predictions using a trained regression model.
    
    Args:
        model (LinearRegression): A fitted regression model.
        X (np.ndarray): The data to make predictions on.
        
    Returns:
        np.ndarray: The predicted values.
    """
    return model.predict(X)

def calculate_regression_metrics(y_true, y_pred):
    """
    Calculates a dictionary of common regression metrics.
    
    Args:
        y_true (np.ndarray): The true target values.
        y_pred (np.ndarray): The predicted values from the model.
        
    Returns:
        dict: A dictionary containing MSE, RMSE, MAPE, and R2 Score.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Handle MAPE calculation to avoid division by zero
    try:
        mape = mean_absolute_percentage_error(y_true, y_pred)
    except:
        # Calculate MAPE manually with handling for zero values
        mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-8))) * 100
    
    r2 = r2_score(y_true, y_pred)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAPE': mape, 'R2 Score': r2}

# CLUSTERING FUNCTIONS (A4, A5, A6, A7)

def perform_kmeans_clustering(X, n_clusters):
    """
    Performs K-Means clustering on the given data.
    
    Args:
        X (np.ndarray): The data to cluster.
        n_clusters (int): The number of clusters (k).
        
    Returns:
        KMeans: The fitted KMeans model object.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init="auto")
    kmeans.fit(X)
    return kmeans

def calculate_clustering_metrics(X, labels):
    """
    Calculates a dictionary of common clustering evaluation metrics.
    
    Args:
        X (np.ndarray): The data that was clustered.
        labels (np.ndarray): The cluster labels assigned by KMeans.
        
    Returns:
        dict: A dictionary containing Silhouette, Calinski-Harabasz, and Davies-Bouldin scores.
    """
    sil_score = silhouette_score(X, labels)
    ch_score = calinski_harabasz_score(X, labels)
    db_score = davies_bouldin_score(X, labels)
    
    return {'Silhouette Score': sil_score, 'Calinski-Harabasz Score': ch_score, 'Davies-Bouldin Score': db_score}

def plot_elbow_method(k_range, inertia_scores):
    """
    Plots the inertia scores against k values to find the "elbow".
    
    Args:
        k_range (list): The list of k values tested.
        inertia_scores (list): The corresponding inertia for each k.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_scores, marker='o', linestyle='--')
    plt.title('Elbow Method for Optimal K', fontsize=16)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia (Within-cluster sum of squares)', fontsize=12)
    plt.xticks(k_range)
    plt.grid(True)
    plt.show()

def plot_clustering_evaluation_scores(k_range, all_scores):
    """
    Plots the different clustering metrics against k values.
    
    Args:
        k_range (list): The list of k values tested.
        all_scores (dict): A dictionary where keys are metric names and values are lists of scores.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    fig.suptitle('Clustering Evaluation Metrics vs. Number of Clusters (k)', fontsize=16)
    
    axes[0].plot(k_range, all_scores['Silhouette'], marker='o')
    axes[0].set_ylabel('Silhouette Score', fontsize=12)
    axes[0].set_title('Higher is better', fontsize=10)
    axes[0].grid(True)
    
    axes[1].plot(k_range, all_scores['Calinski-Harabasz'], marker='o')
    axes[1].set_ylabel('Calinski-Harabasz Score', fontsize=12)
    axes[1].set_title('Higher is better', fontsize=10)
    axes[1].grid(True)

    axes[2].plot(k_range, all_scores['Davies-Bouldin'], marker='o')
    axes[2].set_ylabel('Davies-Bouldin Score', fontsize=12)
    axes[2].set_title('Lower is better', fontsize=10)
    axes[2].grid(True)
    
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.xticks(k_range)
    plt.show()


# MAIN EXECUTION BLOCK

if __name__ == "__main__":
    
    # --- Data Loading and Preparation ---
    print("Loading and preparing data (first 500 rows for faster processing)...")
    try:
        X_features, y_labels, original_df, feature_names, scaler = load_and_prepare_data(CSV_FILE_PATH, nrows=500)
    except FileNotFoundError:
        print(f"Error: Could not find the file '{CSV_FILE_PATH}'. Please make sure the file exists in the current directory.")
        exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        exit(1)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y_labels, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Data split into {len(X_train)} training samples and {len(X_test)} testing samples.")
    
    # --- Regression Problem Setup ---
    # For simple regression (A1, A2): Use Age to predict Stress_Level
    age_index = feature_names.index('Age') if 'Age' in feature_names else 0
    X_train_reg_simple = X_train[:, [age_index]]  # Use Age as predictor
    X_test_reg_simple = X_test[:, [age_index]]
    
    # For multiple regression (A3): Use all features except the target
    X_train_reg_multi = X_train  # All features
    X_test_reg_multi = X_test
    
    y_train_reg_target = y_train  # Stress_Level as target
    y_test_reg_target = y_test

    # --- A1 & A2: Simple Linear Regression (One Attribute) ---
    print("\n\n" + "="*50)
    print("--- Task A1 & A2: Simple Linear Regression (Age -> Stress Level) ---")
    print("="*50)
    
    # Train model
    simple_reg_model = train_linear_regression(X_train_reg_simple, y_train_reg_target)
    
    # Predict on training data
    y_train_pred_simple = get_regression_predictions(simple_reg_model, X_train_reg_simple)
    train_metrics_simple = calculate_regression_metrics(y_train_reg_target, y_train_pred_simple)
    
    # Predict on test data
    y_test_pred_simple = get_regression_predictions(simple_reg_model, X_test_reg_simple)
    test_metrics_simple = calculate_regression_metrics(y_test_reg_target, y_test_pred_simple)
    
    print("\nMetrics on TRAINING data:")
    for metric, value in train_metrics_simple.items():
        print(f"  - {metric}: {value:.4f}")
        
    print("\nMetrics on TEST data:")
    for metric, value in test_metrics_simple.items():
        print(f"  - {metric}: {value:.4f}")

    # --- A3: Multiple Linear Regression (All Attributes) ---
    print("\n\n" + "="*50)
    print("--- Task A3: Multiple Linear Regression (All Features -> Stress Level) ---")
    print("="*50)
    
    # Train model
    multi_reg_model = train_linear_regression(X_train_reg_multi, y_train_reg_target)
    
    # Predict on training data
    y_train_pred_multi = get_regression_predictions(multi_reg_model, X_train_reg_multi)
    train_metrics_multi = calculate_regression_metrics(y_train_reg_target, y_train_pred_multi)
    
    # Predict on test data
    y_test_pred_multi = get_regression_predictions(multi_reg_model, X_test_reg_multi)
    test_metrics_multi = calculate_regression_metrics(y_test_reg_target, y_test_pred_multi)
    
    print("\nMetrics on TRAINING data:")
    for metric, value in train_metrics_multi.items():
        print(f"  - {metric}: {value:.4f}")
        
    print("\nMetrics on TEST data:")
    for metric, value in test_metrics_multi.items():
        print(f"  - {metric}: {value:.4f}")

    # Display feature importance (coefficients)
    print(f"\nFeature coefficients (importance):")
    for i, (feature, coef) in enumerate(zip(feature_names, multi_reg_model.coef_)):
        print(f"  - {feature}: {coef:.4f}")

    # --- A4 & A5: K-Means Clustering (k=2) & Evaluation ---
    print("\n\n" + "="*50)
    print("--- Task A4 & A5: K-Means Clustering with k=2 ---")
    print("="*50)
    
    # Perform clustering
    kmeans_k2 = perform_kmeans_clustering(X_train, n_clusters=2)
    k2_labels = kmeans_k2.labels_
    k2_cluster_centers = kmeans_k2.cluster_centers_
    
    print(f"Cluster centers shape: {k2_cluster_centers.shape}")
    print(f"Number of samples in cluster 0: {np.sum(k2_labels == 0)}")
    print(f"Number of samples in cluster 1: {np.sum(k2_labels == 1)}")
    
    # Evaluate the clustering
    k2_metrics = calculate_clustering_metrics(X_train, k2_labels)
    
    print("\nEvaluation metrics for k=2 clustering:")
    for metric, value in k2_metrics.items():
        print(f"  - {metric}: {value:.4f}")

    # --- A6 & A7: Finding Optimal K ---
    print("\n\n" + "="*50)
    print("--- Task A6 & A7: Finding Optimal K for Clustering ---")
    print("="*50)
    
    k_range = list(range(2, 11))  # Test k from 2 to 10
    inertia_scores = []
    clustering_scores = {'Silhouette': [], 'Calinski-Harabasz': [], 'Davies-Bouldin': []}
    
    print("Calculating clustering metrics for k from 2 to 10...")
    for k in k_range:
        kmeans_model = perform_kmeans_clustering(X_train, n_clusters=k)
        labels = kmeans_model.labels_
        
        # A7: Elbow method
        inertia_scores.append(kmeans_model.inertia_)
        
        # A6: Other metrics
        metrics = calculate_clustering_metrics(X_train, labels)
        clustering_scores['Silhouette'].append(metrics['Silhouette Score'])
        clustering_scores['Calinski-Harabasz'].append(metrics['Calinski-Harabasz Score'])
        clustering_scores['Davies-Bouldin'].append(metrics['Davies-Bouldin Score'])
    
    print("Calculations complete. Generating plots...")
    
    # A7 Plot
    plot_elbow_method(k_range, inertia_scores)
    
    # A6 Plot
    plot_clustering_evaluation_scores(k_range, clustering_scores)

    # Print summary of optimal k suggestions
    print("\n" + "="*50)
    print("SUMMARY - Optimal K Suggestions:")
    print("="*50)
    
    # Find optimal k for each metric
    best_silhouette_k = k_range[np.argmax(clustering_scores['Silhouette'])]
    best_ch_k = k_range[np.argmax(clustering_scores['Calinski-Harabasz'])]
    best_db_k = k_range[np.argmin(clustering_scores['Davies-Bouldin'])]
    
    print(f"Best k according to Silhouette Score: {best_silhouette_k}")
    print(f"Best k according to Calinski-Harabasz Score: {best_ch_k}")
    print(f"Best k according to Davies-Bouldin Score: {best_db_k}")
    print("\nNote: For Elbow method, look for the 'elbow' point in the plot where the rate of decrease slows down.")
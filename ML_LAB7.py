import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
import warnings
warnings.filterwarnings('ignore')

# For XGBoost and CatBoost (install if needed: pip install xgboost catboost)
try:
    from xgboost import XGBClassifier
    xgb_available = True
    print("✓ XGBoost available")
except ImportError:
    print("⚠ XGBoost not available. Install with: pip install xgboost")
    xgb_available = False

try:
    from catboost import CatBoostClassifier
    catboost_available = True
    print("✓ CatBoost available")
except ImportError:
    print("⚠ CatBoost not available. Install with: pip install catboost")
    catboost_available = False

# For SHAP and LIME (Optional section)
try:
    import shap
    shap_available = True
    print("✓ SHAP available")
except ImportError:
    print("⚠ SHAP not available. Install with: pip install shap")
    shap_available = False

try:
    import lime
    from lime.lime_base import LimeBase
    from lime.lime_tabular import LimeTabularExplainer
    lime_available = True
    print("✓ LIME available")
except ImportError:
    print("⚠ LIME not available. Install with: pip install lime")
    lime_available = False

# A1. Load and preprocess the data
def load_and_preprocess_data():
    """Load and preprocess the mental health dataset"""
    # Load only the first 500 rows for faster processing
    print("Loading first 500 rows from MHDS.csv for faster processing...")
    df = pd.read_csv('MHDS.csv', nrows=500)
    
    print("Dataset Info:")
    print(f"Shape: {df.shape}")
    print("\nColumn names:")
    print(df.columns.tolist())
    print("\nFirst few rows:")
    print(df.head())
    
    # Handle missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Encode categorical variables
    le = LabelEncoder()
    categorical_cols = ['Gender', 'Occupation', 'Country', 'Mental_Health_Condition', 
                       'Severity', 'Consultation_History', 'Stress_Level', 
                       'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption', 'Medication_Usage']
    
    df_encoded = df.copy()
    label_encoders = {}
    
    for col in categorical_cols:
        if col in df_encoded.columns:
            label_encoders[col] = LabelEncoder()
            df_encoded[col] = label_encoders[col].fit_transform(df_encoded[col].astype(str))
    
    return df_encoded, label_encoders, df

def prepare_features_target(df_encoded):
    """Prepare features and target variable"""
    # Assuming 'Mental_Health_Condition' is our target variable
    target_col = 'Mental_Health_Condition'
    
    if target_col in df_encoded.columns:
        X = df_encoded.drop([target_col, 'User_ID'], axis=1)  # Remove ID and target
        y = df_encoded[target_col]
    else:
        print(f"Target column '{target_col}' not found. Available columns:")
        print(df_encoded.columns.tolist())
        return None, None
    
    return X, y

# A2. Cross-validation with RandomizedSearchCV
def hyperparameter_tuning(X_train, y_train):
    """Perform hyperparameter tuning using RandomizedSearchCV"""
    
    # Define parameter grids for different models
    param_grids = {
        'Perceptron': {
            'alpha': [0.0001, 0.001, 0.01, 0.1],
            'max_iter': [100, 500, 1000],
            'penalty': ['l2', 'l1', 'elasticnet']
        },
        'SVM': {
            'C': [0.1, 1, 10, 100],
            'kernel': ['linear', 'rbf', 'poly'],
            'gamma': ['scale', 'auto', 0.001, 0.01]
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        },
        'DecisionTree': {
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'criterion': ['gini', 'entropy']
        }
    }
    
    models = {
        'Perceptron': Perceptron(random_state=42),
        'SVM': SVC(random_state=42),
        'RandomForest': RandomForestClassifier(random_state=42),
        'DecisionTree': DecisionTreeClassifier(random_state=42)
    }
    
    best_models = {}
    
    for name, model in models.items():
        print(f"\nTuning {name}...")
        
        random_search = RandomizedSearchCV(
            model, 
            param_grids[name], 
            n_iter=10,  # Reduced iterations for faster processing
            cv=3,       # Reduced CV folds for faster processing  
            scoring='accuracy', 
            random_state=42,
            n_jobs=-1
        )
        
        random_search.fit(X_train, y_train)
        
        best_models[name] = random_search.best_estimator_
        print(f"Best parameters for {name}: {random_search.best_params_}")
        print(f"Best CV score for {name}: {random_search.best_score_:.4f}")
    
    return best_models

# A3. Classification with multiple algorithms
def train_classifiers(X_train, X_test, y_train, y_test, best_models=None):
    """Train various classifiers and evaluate performance"""
    
    # Initialize classifiers
    classifiers = {
        'Perceptron': Perceptron(random_state=42),
        'SVM': SVC(random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(random_state=42),
        'AdaBoost': AdaBoostClassifier(random_state=42),
        'Naive Bayes': GaussianNB(),
        'MLP': MLPClassifier(random_state=42, max_iter=500)
    }
    
    # Add XGBoost and CatBoost if available
    if xgb_available:
        classifiers['XGBoost'] = XGBClassifier(random_state=42, eval_metric='logloss')
    
    if catboost_available:
        classifiers['CatBoost'] = CatBoostClassifier(random_state=42, verbose=False)
    
    # Use best models from hyperparameter tuning if available
    if best_models:
        for name, model in best_models.items():
            classifiers[name] = model
    
    results = []
    trained_models = {}
    
    for name, clf in classifiers.items():
        print(f"\nTraining {name}...")
        
        # Train the model
        clf.fit(X_train, y_train)
        
        # Make predictions
        y_train_pred = clf.predict(X_train)
        y_test_pred = clf.predict(X_test)
        
        # Calculate metrics
        train_accuracy = accuracy_score(y_train, y_train_pred)
        test_accuracy = accuracy_score(y_test, y_test_pred)
        train_f1 = f1_score(y_train, y_train_pred, average='weighted')
        test_f1 = f1_score(y_test, y_test_pred, average='weighted')
        train_precision = precision_score(y_train, y_train_pred, average='weighted')
        test_precision = precision_score(y_test, y_test_pred, average='weighted')
        train_recall = recall_score(y_train, y_train_pred, average='weighted')
        test_recall = recall_score(y_test, y_test_pred, average='weighted')
        
        results.append({
            'Algorithm': name,
            'Train_Accuracy': train_accuracy,
            'Test_Accuracy': test_accuracy,
            'Train_F1': train_f1,
            'Test_F1': test_f1,
            'Train_Precision': train_precision,
            'Test_Precision': test_precision,
            'Train_Recall': train_recall,
            'Test_Recall': test_recall,
            'Overfitting': train_accuracy - test_accuracy
        })
        
        trained_models[name] = clf
        
        print(f"{name} - Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")
    
    return pd.DataFrame(results), trained_models

# Optional Section - SHAP Implementation
def shap_analysis(model, X_train, X_test, feature_names):
    """Perform SHAP analysis for model interpretability"""
    if not shap_available:
        print("SHAP not available. Please install with: pip install shap")
        return
    
    print("\nPerforming SHAP Analysis...")
    
    try:
        # For tree-based models, use TreeExplainer
        if hasattr(model, 'estimators_') or 'forest' in str(type(model)).lower():
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test[:50])
            
            plt.figure(figsize=(12, 8))
            if len(shap_values) == 2:  # Binary classification
                shap.summary_plot(shap_values[1], X_test[:50], feature_names=feature_names, show=False)
            else:
                shap.summary_plot(shap_values, X_test[:50], feature_names=feature_names, show=False)
            plt.title("SHAP Feature Importance Summary")
            plt.tight_layout()
            plt.savefig('shap_summary.png', dpi=300, bbox_inches='tight')
            plt.show()
            
        else:
            print("SHAP analysis skipped - model type not supported for this demo")
            return None
            
    except Exception as e:
        print(f"SHAP analysis failed: {e}")
        print("This is normal - SHAP can be tricky with some model types")
        return None
    
    return shap_values

# Optional Section - LIME Implementation
def lime_analysis(model, X_train, X_test, feature_names, class_names):
    """Perform LIME analysis for model interpretability"""
    if not lime_available:
        print("LIME not available. Please install with: pip install lime")
        return
    
    print("\nPerforming LIME Analysis...")
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train.values,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # Explain a few instances
    for i in range(min(3, len(X_test))):
        exp = explainer.explain_instance(
            X_test.iloc[i].values, 
            model.predict_proba, 
            num_features=len(feature_names)
        )
        
        print(f"\nLIME Explanation for instance {i}:")
        for feature, importance in exp.as_list():
            print(f"{feature}: {importance:.4f}")

def visualize_results(results_df):
    """Create visualizations for model comparison"""
    
    # Set up the plotting style
    plt.style.use('default')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    ax1 = axes[0, 0]
    x = range(len(results_df))
    width = 0.35
    ax1.bar([i - width/2 for i in x], results_df['Train_Accuracy'], width, label='Train', alpha=0.8)
    ax1.bar([i + width/2 for i in x], results_df['Test_Accuracy'], width, label='Test', alpha=0.8)
    ax1.set_xlabel('Algorithms')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Train vs Test Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(results_df['Algorithm'], rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # F1 Score comparison
    ax2 = axes[0, 1]
    ax2.bar([i - width/2 for i in x], results_df['Train_F1'], width, label='Train', alpha=0.8)
    ax2.bar([i + width/2 for i in x], results_df['Test_F1'], width, label='Test', alpha=0.8)
    ax2.set_xlabel('Algorithms')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Train vs Test F1 Score Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(results_df['Algorithm'], rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Overfitting analysis
    ax3 = axes[1, 0]
    bars = ax3.bar(results_df['Algorithm'], results_df['Overfitting'], alpha=0.8)
    ax3.set_xlabel('Algorithms')
    ax3.set_ylabel('Overfitting (Train - Test Accuracy)')
    ax3.set_title('Overfitting Analysis')
    ax3.tick_params(axis='x', rotation=45)
    ax3.grid(True, alpha=0.3)
    
    # Color bars based on overfitting level
    for i, bar in enumerate(bars):
        if results_df['Overfitting'].iloc[i] > 0.1:
            bar.set_color('red')
        elif results_df['Overfitting'].iloc[i] > 0.05:
            bar.set_color('orange')
        else:
            bar.set_color('green')
    
    # Performance metrics heatmap
    ax4 = axes[1, 1]
    metrics_data = results_df[['Test_Accuracy', 'Test_F1', 'Test_Precision', 'Test_Recall']].T
    metrics_data.columns = results_df['Algorithm']
    sns.heatmap(metrics_data, annot=True, fmt='.3f', cmap='Blues', ax=ax4)
    ax4.set_title('Test Performance Metrics Heatmap')
    ax4.set_ylabel('Metrics')
    
    plt.tight_layout()
    
    # Save the figure
    plt.savefig('model_comparison_results.png', dpi=300, bbox_inches='tight')
    print("✓ Main results visualization saved as 'model_comparison_results.png'")
    plt.show()

def check_file_exists():
    """Check if the CSV file exists"""
    import os
    if not os.path.exists('MHDS.csv'):
        print("ERROR: MHDS.csv file not found in current directory!")
        print("Please ensure the CSV file is in the same folder as this script.")
        print(f"Current directory: {os.getcwd()}")
        return False
    return True

def main():
    """Main function to run the complete analysis"""
    
    print("=" * 60)
    print("MENTAL HEALTH CLASSIFICATION ANALYSIS - LAB 07")
    print("=" * 60)
    
    # Check if file exists
    if not check_file_exists():
        return None, None, None
    
    # A1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df_encoded, label_encoders, df_original = load_and_preprocess_data()
    
    # Prepare features and target
    X, y = prepare_features_target(df_encoded)
    
    if X is None or y is None:
        print("Error in data preparation. Exiting...")
        return
    
    print(f"\nFeature shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target distribution:\n{pd.Series(y).value_counts()}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), 
        columns=X_test.columns, 
        index=X_test.index
    )
    
    # A2. Hyperparameter tuning
    print("\n2. Performing hyperparameter tuning...")
    best_models = hyperparameter_tuning(X_train_scaled, y_train)
    
    # A3. Train classifiers
    print("\n3. Training multiple classifiers...")
    results_df, trained_models = train_classifiers(
        X_train_scaled, X_test_scaled, y_train, y_test, best_models
    )
    
    # Display results table
    print("\n4. RESULTS SUMMARY:")
    print("=" * 120)
    print(results_df.round(4))
    
    # Find best performing model
    best_model_name = results_df.loc[results_df['Test_Accuracy'].idxmax(), 'Algorithm']
    best_model = trained_models[best_model_name]
    print(f"\nBest performing model: {best_model_name}")
    print(f"Test Accuracy: {results_df.loc[results_df['Test_Accuracy'].idxmax(), 'Test_Accuracy']:.4f}")
    
    # Visualize results
    print("\n5. Creating visualizations...")
    visualize_results(results_df)
    
    # Optional: SHAP Analysis
    if shap_available:
        print("\n6. SHAP Analysis (Optional)...")
        try:
            shap_values = shap_analysis(best_model, X_train_scaled, X_test_scaled, X.columns)
        except Exception as e:
            print(f"SHAP analysis failed: {e}")
    
    # Optional: LIME Analysis
    if lime_available:
        print("\n7. LIME Analysis (Optional)...")
        try:
            class_names = [str(i) for i in sorted(y.unique())]
            lime_analysis(best_model, X_train_scaled, X_test_scaled, X.columns, class_names)
        except Exception as e:
            print(f"LIME analysis failed: {e}")
    
    # Generate detailed classification report for best model
    print(f"\n8. Detailed Classification Report for {best_model_name}:")
    print("=" * 60)
    y_pred = best_model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # Save confusion matrix
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Confusion matrix saved as 'confusion_matrix.png'")
    plt.show()
    
    print("\n9. OBSERVATIONS AND INSIGHTS:")
    print("=" * 50)
    
    # Analysis of results
    best_accuracy = results_df['Test_Accuracy'].max()
    worst_accuracy = results_df['Test_Accuracy'].min()
    avg_accuracy = results_df['Test_Accuracy'].mean()
    
    print(f"• Best Test Accuracy: {best_accuracy:.4f} ({best_model_name})")
    print(f"• Worst Test Accuracy: {worst_accuracy:.4f}")
    print(f"• Average Test Accuracy: {avg_accuracy:.4f}")
    
    # Overfitting analysis
    high_overfitting = results_df[results_df['Overfitting'] > 0.1]
    if not high_overfitting.empty:
        print(f"• Models with high overfitting (>10%): {', '.join(high_overfitting['Algorithm'].tolist())}")
    
    # Best balanced model (good accuracy, low overfitting)
    results_df['Balance_Score'] = results_df['Test_Accuracy'] - results_df['Overfitting']
    best_balanced = results_df.loc[results_df['Balance_Score'].idxmax(), 'Algorithm']
    print(f"• Most balanced model (accuracy vs overfitting): {best_balanced}")
    
    print(f"\n• Dataset size: 500 samples (subset) with {X.shape[1]} features")
    print(f"• Original dataset had ~50k rows, using first 500 for demonstration")
    print(f"• Class distribution shows potential imbalance: {dict(pd.Series(y).value_counts())}")
    
    return results_df, trained_models, best_model

if __name__ == "__main__":
    # Run the complete analysis
    results, models, best_model = main()
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETED SUCCESSFULLY!")
    print("=" * 60)
    print("\nNext Steps for Report:")
    print("1. Document the methodology used")
    print("2. Include the results table and visualizations")
    print("3. Discuss observations about model performance")
    print("4. Analyze feature importance from SHAP/LIME")
    print("5. Conclude with recommendations for deployment")
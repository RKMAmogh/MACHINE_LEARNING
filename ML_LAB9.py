import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, ExtraTreesClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, precision_recall_fscore_support)
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import lime
from lime import lime_tabular
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(filepath):
    """
    Load the mental health dataset and perform initial preprocessing
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        tuple: Processed features and target variable
    """
    # Load data
    df = pd.read_csv(filepath)
    
    # Display basic information about the dataset
    print("Dataset Shape:", df.shape)
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    # Define target variable (Mental_Health_Condition)
    target = 'Mental_Health_Condition'
    
    # Convert target to binary (Yes=1, No=0)
    df[target] = df[target].map({'Yes': 1, 'No': 0})
    
    # Separate features and target
    X = df.drop([target, 'User_ID'], axis=1)  # Remove User_ID as it's not predictive
    y = df[target]
    
    return X, y, df

def create_preprocessing_pipeline():
    """
    Create preprocessing pipeline for numerical and categorical features
    
    Returns:
        ColumnTransformer: Preprocessing pipeline
    """
    # Define numerical and categorical columns
    # Note: Stress_Level appears to be categorical based on the data
    numerical_features = ['Age', 'Sleep_Hours', 'Work_Hours', 
                         'Physical_Activity_Hours', 'Social_Media_Usage']
    categorical_features = ['Gender', 'Occupation', 'Country', 'Severity', 
                          'Consultation_History', 'Stress_Level', 'Diet_Quality', 
                          'Smoking_Habit', 'Alcohol_Consumption', 'Medication_Usage']
    
    # Create preprocessing steps
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='unknown')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    return preprocessor

def create_base_models():
    """
    Create a list of base models for stacking ensemble
    
    Returns:
        list: List of (name, model) tuples
    """
    base_models = [
        ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
        ('gb', GradientBoostingClassifier(n_estimators=100, random_state=42)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=42)),
        ('knn', KNeighborsClassifier(n_neighbors=5)),
        ('nb', GaussianNB()),
        ('et', ExtraTreesClassifier(n_estimators=100, random_state=42))
    ]
    
    return base_models

def create_stacking_classifier(base_models):
    """
    Create stacking classifier with different meta-models
    
    Args:
        base_models (list): List of base models
    
    Returns:
        dict: Dictionary of stacking classifiers with different meta-models
    """
    stacking_models = {}
    
    # Different meta-models to experiment with
    meta_models = {
        'logistic': LogisticRegression(random_state=42, max_iter=1000),
        'rf_meta': RandomForestClassifier(n_estimators=50, random_state=42),
        'gb_meta': GradientBoostingClassifier(n_estimators=50, random_state=42)
    }
    
    for meta_name, meta_model in meta_models.items():
        stacking_models[f'stacking_{meta_name}'] = StackingClassifier(
            estimators=base_models,
            final_estimator=meta_model,
            cv=5,
            n_jobs=-1
        )
    
    return stacking_models

def create_complete_pipeline(preprocessor, model):
    """
    Create complete pipeline with preprocessing and model
    
    Args:
        preprocessor: Preprocessing pipeline
        model: Machine learning model
    
    Returns:
        Pipeline: Complete pipeline
    """
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    return pipeline

def evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name):
    """
    Evaluate model performance and return metrics
    
    Args:
        pipeline: Trained pipeline
        X_train, X_test, y_train, y_test: Train/test splits
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary containing evaluation metrics
    """
    # Make predictions
    y_pred = pipeline.predict(X_test)
    y_pred_proba = pipeline.predict_proba(X_test)[:, 1] if hasattr(pipeline, 'predict_proba') else None
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # Cross-validation score
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std()
    }
    
    return results, y_pred

def create_lime_explainer(pipeline, X_train, feature_names):
    """
    Create LIME explainer for pipeline interpretation
    
    Args:
        pipeline: Trained pipeline
        X_train: Training data
        feature_names: List of feature names
    
    Returns:
        LimeTabularExplainer: LIME explainer object
    """
    # Get preprocessed training data for LIME
    X_train_processed = pipeline.named_steps['preprocessor'].fit_transform(X_train)
    
    # Get feature names after preprocessing
    preprocessor = pipeline.named_steps['preprocessor']
    
    # Get transformed feature names
    try:
        # Try to get feature names from the preprocessor
        if hasattr(preprocessor, 'get_feature_names_out'):
            feature_names_processed = preprocessor.get_feature_names_out()
        else:
            # Fallback: create generic feature names
            feature_names_processed = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    except:
        # Final fallback
        feature_names_processed = [f'feature_{i}' for i in range(X_train_processed.shape[1])]
    
    # Create LIME explainer
    explainer = lime_tabular.LimeTabularExplainer(
        X_train_processed,
        feature_names=feature_names_processed,
        class_names=['No Mental Health Condition', 'Mental Health Condition'],
        mode='classification'
    )
    
    return explainer

def explain_predictions(explainer, pipeline, X_test, instance_idx=0):
    """
    Generate LIME explanations for specific instances
    
    Args:
        explainer: LIME explainer object
        pipeline: Trained pipeline
        X_test: Test data
        instance_idx (int): Index of instance to explain
    
    Returns:
        LIME explanation object
    """
    # Preprocess the test instance
    X_test_processed = pipeline.named_steps['preprocessor'].transform(X_test)
    
    # Create explanation
    explanation = explainer.explain_instance(
        X_test_processed[instance_idx],
        pipeline.named_steps['classifier'].predict_proba,
        num_features=10
    )
    
    return explanation

def plot_results_comparison(results_df):
    """
    Create visualization comparing model performance
    
    Args:
        results_df (DataFrame): Results dataframe
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    axes[0, 0].bar(results_df['Model'], results_df['Accuracy'])
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    axes[0, 1].bar(results_df['Model'], results_df['F1-Score'])
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Precision vs Recall
    axes[1, 0].scatter(results_df['Precision'], results_df['Recall'])
    for i, model in enumerate(results_df['Model']):
        axes[1, 0].annotate(model, (results_df['Precision'].iloc[i], results_df['Recall'].iloc[i]))
    axes[1, 0].set_title('Precision vs Recall')
    axes[1, 0].set_xlabel('Precision')
    axes[1, 0].set_ylabel('Recall')
    
    # Cross-validation scores with error bars
    axes[1, 1].bar(results_df['Model'], results_df['CV_Mean'], 
                   yerr=results_df['CV_Std'], capsize=5)
    axes[1, 1].set_title('Cross-Validation Scores')
    axes[1, 1].set_ylabel('CV Mean Accuracy')
    axes[1, 1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the complete analysis
    """
    print("="*60)
    print("MENTAL HEALTH DATA ANALYSIS - LAB 09")
    print("Stacking Ensemble Methods and Pipeline Implementation")
    print("="*60)
    
    # Load and preprocess data
    print("\n1. Loading and Preprocessing Data...")
    X, y, df = load_and_preprocess_data('MHDS.csv')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Create preprocessing pipeline
    print("\n2. Creating Preprocessing Pipeline...")
    preprocessor = create_preprocessing_pipeline()
    
    # Create base models
    print("\n3. Creating Base Models...")
    base_models = create_base_models()
    
    # Create stacking classifiers
    print("\n4. Creating Stacking Classifiers...")
    stacking_models = create_stacking_classifier(base_models)
    
    # Combine all models for evaluation
    all_models = dict(base_models)
    all_models.update(stacking_models)
    
    # Evaluate all models
    print("\n5. Training and Evaluating Models...")
    results = []
    trained_pipelines = {}
    
    for model_name, model in all_models.items():
        print(f"Training {model_name}...")
        
        # Create complete pipeline
        pipeline = create_complete_pipeline(preprocessor, model)
        
        # Train pipeline
        pipeline.fit(X_train, y_train)
        
        # Evaluate model
        result, y_pred = evaluate_model(pipeline, X_train, X_test, y_train, y_test, model_name)
        results.append(result)
        trained_pipelines[model_name] = pipeline
        
        print(f"{model_name} - Accuracy: {result['Accuracy']:.4f}, F1: {result['F1-Score']:.4f}")
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Accuracy', ascending=False)
    
    print("\n6. Model Performance Summary:")
    print(results_df.to_string(index=False, float_format='%.4f'))
    
    # Visualize results
    print("\n7. Creating Performance Visualizations...")
    plot_results_comparison(results_df)
    
    # Select best model for LIME explanation
    best_model_name = results_df.iloc[0]['Model']
    best_pipeline = trained_pipelines[best_model_name]
    
    print(f"\n8. LIME Explanation for Best Model: {best_model_name}")
    
    # Create LIME explainer
    try:
        explainer = create_lime_explainer(best_pipeline, X_train, X.columns.tolist())
        
        # Explain a few predictions
        for i in [0, 1, 2]:
            if i < len(X_test):
                print(f"\nExplaining instance {i}:")
                explanation = explain_predictions(explainer, best_pipeline, X_test, i)
                
                # Get actual prediction
                actual_pred = best_pipeline.predict(X_test.iloc[[i]])[0]
                actual_proba = best_pipeline.predict_proba(X_test.iloc[[i]])[0]
                
                print(f"Actual prediction: {'Mental Health Condition' if actual_pred == 1 else 'No Mental Health Condition'}")
                print(f"Prediction probability: {actual_proba[1]:.4f}")
                
                # Print top features from explanation
                exp_list = explanation.as_list()
                print("Top contributing features:")
                for feature, weight in exp_list[:5]:
                    print(f"  {feature}: {weight:.4f}")
                
    except Exception as e:
        print(f"LIME explanation encountered an issue: {str(e)}")
        print("Continuing with basic model interpretation...")
        
        # Alternative: Show feature importance for tree-based models
        if hasattr(best_pipeline.named_steps['classifier'], 'feature_importances_'):
            print("\nFeature Importance from best model:")
            # Get feature names after preprocessing
            try:
                feature_names = best_pipeline.named_steps['preprocessor'].get_feature_names_out()
                importances = best_pipeline.named_steps['classifier'].feature_importances_
                
                # Sort by importance
                feature_importance = list(zip(feature_names, importances))
                feature_importance.sort(key=lambda x: x[1], reverse=True)
                
                print("Top 10 most important features:")
                for i, (feature, importance) in enumerate(feature_importance[:10]):
                    print(f"{i+1}. {feature}: {importance:.4f}")
                    
            except Exception as fe:
                print(f"Could not extract feature importance: {str(fe)}")
    
    print("\n9. Analysis Complete!")
    print("="*60)
    
    return results_df, trained_pipelines, best_pipeline

# Execute main function
if __name__ == "__main__":
    results_df, trained_pipelines, best_pipeline = main()
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from typing import Union, Optional, Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Load the dataset - using first 500 records from MHDS.csv
try:
    df_full = pd.read_csv('MHDS.csv')
    df = df_full.head(500).copy()  # Use only first 500 records
    print(f"Successfully loaded dataset! Using {len(df)} records from {len(df_full)} total records.")
except FileNotFoundError:
    print("MHDS.csv file not found. Please ensure the file is in the current directory.")
    exit()

print(f"Dataset shape: {df.shape}")
print("\nDataset columns:")
print(df.columns.tolist())
print("\nDataset preview:")
print(df.head())

print("\nDataset info:")
print(df.info())

# A1: Calculate Entropy Function
def calculate_entropy(labels):

    if len(labels) == 0:
        return 0
    
    # Count occurrences of each label
    value_counts = Counter(labels)
    total_samples = len(labels)
    
    entropy = 0
    for count in value_counts.values():
        probability = count / total_samples
        if probability > 0:
            entropy -= probability * np.log2(probability)
    
    return entropy

# Equal width binning function for continuous variables
def equal_width_binning(data, n_bins=4):

    min_val, max_val = np.min(data), np.max(data)
    bin_width = (max_val - min_val) / n_bins
    
    bins = []
    for i in range(n_bins):
        bins.append(min_val + i * bin_width)
    bins.append(max_val)
    
    # Create labels for bins
    labels = [f'Bin_{i+1}' for i in range(n_bins)]
    
    # Bin the data
    binned_data = pd.cut(data, bins=bins, labels=labels, include_lowest=True)
    
    return binned_data, bins

# Test entropy calculation with Mental_Health_Condition as target
target_variable = 'Mental_Health_Condition'
if target_variable in df.columns:
    entropy_value = calculate_entropy(df[target_variable])
    print(f"\nA1: Entropy of {target_variable}: {entropy_value:.4f}")
    
    # Show distribution of target variable
    print(f"\nDistribution of {target_variable}:")
    print(df[target_variable].value_counts())

# A2: Calculate Gini Index
def calculate_gini_index(labels):

    if len(labels) == 0:
        return 0
    
    # Count occurrences of each label
    value_counts = Counter(labels)
    total_samples = len(labels)
    
    gini = 1.0
    for count in value_counts.values():
        probability = count / total_samples
        gini -= probability ** 2
    
    return gini

# Calculate Gini index for target variable
gini_value = calculate_gini_index(df[target_variable])
print(f"\nA2: Gini Index of {target_variable}: {gini_value:.4f}")

# A3: Information Gain Calculation and Root Node Detection
def calculate_information_gain(data, feature, target):
    # Calculate entropy of the entire dataset
    total_entropy = calculate_entropy(data[target])
    
    # Calculate weighted entropy for each unique value of the feature
    feature_values = data[feature].unique()
    weighted_entropy = 0
    
    for value in feature_values:
        subset = data[data[feature] == value]
        subset_entropy = calculate_entropy(subset[target])
        weight = len(subset) / len(data)
        weighted_entropy += weight * subset_entropy
    
    # Information gain = Total entropy - Weighted entropy
    information_gain = total_entropy - weighted_entropy
    return information_gain

def find_best_root_feature(data, features, target):
    best_feature = None
    best_gain = -1
    
    gains = {}
    for feature in features:
        gain = calculate_information_gain(data, feature, target)
        gains[feature] = gain
        
        if gain > best_gain:
            best_gain = gain
            best_feature = feature
    
    return best_feature, gains

# Select categorical features for analysis
categorical_features = ['Gender', 'Occupation', 'Country', 'Severity', 'Consultation_History', 
                       'Stress_Level', 'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption', 
                       'Medication_Usage']

# Filter features that exist in the dataset
available_features = [f for f in categorical_features if f in df.columns]

print(f"\nA3: Available categorical features: {available_features}")

# Find best root node feature
best_feature, all_gains = find_best_root_feature(df, available_features, target_variable)

print(f"\nInformation gains for each feature:")
for feature, gain in sorted(all_gains.items(), key=lambda x: x[1], reverse=True):
    print(f"{feature}: {gain:.4f}")

print(f"\nBest root node feature: {best_feature} (Information Gain: {all_gains[best_feature]:.4f})")

# A4: Binning for continuous features
def binning_function(data, n_bins=4, method='equal_width'):

    if method == 'equal_width':
        return equal_width_binning(data, n_bins)
    elif method == 'equal_frequency':
        # Equal frequency binning using quantiles
        binned_data = pd.qcut(data, q=n_bins, labels=[f'Bin_{i+1}' for i in range(n_bins)], 
                             duplicates='drop')
        return binned_data, None
    else:
        raise ValueError("Method must be 'equal_width' or 'equal_frequency'")

# Test binning on continuous features
continuous_features = ['Age', 'Sleep_Hours', 'Work_Hours', 'Physical_Activity_Hours', 'Social_Media_Usage']
available_continuous = [f for f in continuous_features if f in df.columns]

print(f"\nA4: Binning continuous features: {available_continuous}")

# Create binned versions of continuous features
for feature in available_continuous:
    if feature in df.columns:
        binned_data, bins = binning_function(df[feature], n_bins=4, method='equal_width')
        df[f'{feature}_binned'] = binned_data
        
        print(f"\n{feature} binning:")
        print(f"Original range: {df[feature].min():.2f} - {df[feature].max():.2f}")
        print(f"Binned distribution:")
        print(df[f'{feature}_binned'].value_counts().sort_index())

# A5: Complete Decision Tree Implementation
class DecisionTreeNode:
    def __init__(self, feature=None, value=None, prediction=None, left=None, right=None):
        self.feature = feature
        self.value = value
        self.prediction = prediction
        self.left = left
        self.right = right
        self.is_leaf = prediction is not None

class DecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
    
    def fit(self, X, y, features=None):
        if features is None:
            features = X.columns.tolist()
        
        # Combine X and y for easier handling
        data = X.copy()
        data['target'] = y
        
        self.root = self._build_tree(data, features, 'target', depth=0)
    
    def _build_tree(self, data, features, target, depth):
        """Recursively build the decision tree."""
        
        # Check stopping conditions
        if (depth >= self.max_depth or 
            len(data) < self.min_samples_split or 
            len(data[target].unique()) == 1):
            
            # Return leaf node with most common class
            prediction = data[target].mode().iloc[0]
            return DecisionTreeNode(prediction=prediction)
        
        # Find best feature to split on
        best_feature, _ = find_best_root_feature(data, features, target)
        
        if best_feature is None:
            prediction = data[target].mode().iloc[0]
            return DecisionTreeNode(prediction=prediction)
        
        # Create internal node
        node = DecisionTreeNode(feature=best_feature)
        
        # For categorical features, create child for each unique value
        unique_values = data[best_feature].unique()
        
        if len(unique_values) <= 2:  # Binary split
            # Split into two groups
            most_common_value = data[best_feature].mode().iloc[0]
            
            left_data = data[data[best_feature] == most_common_value]
            right_data = data[data[best_feature] != most_common_value]
            
            if len(left_data) > 0:
                node.left = self._build_tree(left_data, features, target, depth + 1)
            if len(right_data) > 0:
                node.right = self._build_tree(right_data, features, target, depth + 1)
        else:
            # For multi-class categorical, create binary split with most common vs others
            most_common_value = data[best_feature].mode().iloc[0]
            node.value = most_common_value
            
            left_data = data[data[best_feature] == most_common_value]
            right_data = data[data[best_feature] != most_common_value]
            
            if len(left_data) > 0:
                node.left = self._build_tree(left_data, features, target, depth + 1)
            if len(right_data) > 0:
                node.right = self._build_tree(right_data, features, target, depth + 1)
        
        return node
    
    def predict(self, X):
        """Make predictions for input data."""
        predictions = []
        for _, row in X.iterrows():
            pred = self._predict_single(row, self.root)
            predictions.append(pred)
        return np.array(predictions)
    
    def _predict_single(self, row, node):
        """Predict for a single instance."""
        if node.is_leaf:
            return node.prediction
        
        if node.value is not None:
            # Categorical split with specific value
            if row[node.feature] == node.value:
                return self._predict_single(row, node.left) if node.left else node.prediction
            else:
                return self._predict_single(row, node.right) if node.right else node.prediction
        else:
            # Binary split - go left if feature matches most common, right otherwise
            if node.left and hasattr(node.left, 'feature'):
                return self._predict_single(row, node.left)
            elif node.right:
                return self._predict_single(row, node.right)
            else:
                return 'Yes'  # Default prediction

# Build decision tree with selected features
selected_features = ['Stress_Level', 'Diet_Quality', 'Smoking_Habit', 'Alcohol_Consumption']
available_selected = [f for f in selected_features if f in df.columns]

print(f"\nA5: Building Decision Tree with features: {available_selected}")

# Create and train the decision tree
dt = DecisionTree(max_depth=3, min_samples_split=10)
X_train = df[available_selected]
y_train = df[target_variable]

dt.fit(X_train, y_train, available_selected)

# Make predictions
y_pred = dt.predict(X_train)

# Calculate accuracy
accuracy = np.mean(y_pred == y_train) * 100
print(f"Decision Tree Accuracy: {accuracy:.2f}%")

# A6: Visualize Decision Tree
def print_tree(node, depth=0, prefix="Root: "):
    if node is None:
        return
    
    indent = "  " * depth
    if node.is_leaf:
        print(f"{indent}{prefix}Predict: {node.prediction}")
    else:
        print(f"{indent}{prefix}Feature: {node.feature}")
        if node.value is not None:
            print(f"{indent}  Value: {node.value}")
        
        if node.left:
            print_tree(node.left, depth + 1, "Left: ")
        if node.right:
            print_tree(node.right, depth + 1, "Right: ")

print(f"\nA6: Decision Tree Structure:")
print_tree(dt.root)

# A7: Decision Boundary Visualization (2 features)
# Select two features for 2D visualization
if len(available_selected) >= 2:
    feature1, feature2 = available_selected[0], available_selected[1]
    
    print(f"\nA7: Creating decision boundary visualization using {feature1} and {feature2}")
    
    # Create numerical mapping for categorical features
    def create_mapping(series):
        unique_vals = series.unique()
        return {val: i for i, val in enumerate(unique_vals)}
    
    # Map categorical to numerical
    f1_mapping = create_mapping(df[feature1])
    f2_mapping = create_mapping(df[feature2])
    target_mapping = create_mapping(df[target_variable])
    
    # Create numerical data
    X_numeric = np.column_stack([
        df[feature1].map(f1_mapping),
        df[feature2].map(f2_mapping)
    ])
    y_numeric = df[target_variable].map(target_mapping)
    
    # Plot decision boundary
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot
    scatter = plt.scatter(X_numeric[:, 0], X_numeric[:, 1], 
                         c=y_numeric, cmap='viridis', alpha=0.7)
    plt.colorbar(scatter, label=target_variable)
    
    # Set labels with original categorical values
    plt.xlabel(f'{feature1}')
    plt.ylabel(f'{feature2}')
    
    # Create tick labels
    plt.xticks(range(len(f1_mapping)), list(f1_mapping.keys()), rotation=45)
    plt.yticks(range(len(f2_mapping)), list(f2_mapping.keys()))
    
    plt.title(f'Decision Boundary: {feature1} vs {feature2}')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Feature importance analysis
    print(f"\nFeature Analysis:")
    print(f"Information gain for {feature1}: {all_gains.get(feature1, 'N/A'):.4f}")
    print(f"Information gain for {feature2}: {all_gains.get(feature2, 'N/A'):.4f}")

# Summary Statistics
print(f"\n" + "="*50)
print("SUMMARY RESULTS")
print(f"="*50)
print(f"Dataset size: {len(df)} samples")
print(f"Target variable: {target_variable}")
print(f"Entropy: {entropy_value:.4f}")
print(f"Gini Index: {gini_value:.4f}")
print(f"Best root feature: {best_feature}")
print(f"Best information gain: {all_gains[best_feature]:.4f}")
print(f"Decision tree accuracy: {accuracy:.2f}%")
print(f"Tree depth: 3 (max)")
print(f"Features used: {', '.join(available_selected)}")

# Additional analysis for report
print(f"\n" + "="*50)
print("ADDITIONAL ANALYSIS FOR REPORT")
print(f"="*50)

# Class distribution
print(f"\nTarget variable distribution:")
target_dist = df[target_variable].value_counts()
for class_name, count in target_dist.items():
    percentage = (count / len(df)) * 100
    print(f"{class_name}: {count} ({percentage:.1f}%)")

# Feature correlation with target
print(f"\nFeature analysis with target variable:")
for feature in available_selected:
    # Calculate information gain
    gain = calculate_information_gain(df, feature, target_variable)
    print(f"{feature}: Information Gain = {gain:.4f}")

print(f"\n" + "="*50)
print("OBSERVATIONS AND INFERENCES")
print(f"="*50)

# Fix the target distribution analysis
target_counts = target_dist.values
min_count = min(target_counts)
max_count = max(target_counts)
balance_ratio = min_count / max_count

print(f"1. The dataset shows {'balanced' if balance_ratio > 0.4 else 'imbalanced'} class distribution")
print(f"2. {best_feature} is the most informative feature for predicting {target_variable}")
print(f"3. Entropy of {entropy_value:.4f} indicates {'high' if entropy_value > 0.8 else 'moderate' if entropy_value > 0.5 else 'low'} uncertainty in the target variable")
print(f"4. The decision tree achieved {accuracy:.2f}% accuracy on training data")
print(f"5. Features ranked by information gain: {', '.join([f[0] for f in sorted(all_gains.items(), key=lambda x: x[1], reverse=True)[:3]])}")
print(f"6. Class balance ratio: {balance_ratio:.3f} (closer to 1.0 means more balanced)")
print(f"7. Most informative features have low information gain, suggesting complex relationships")
print(f"8. 'Severity' feature shows highest discriminative power with gain of {all_gains[best_feature]:.4f}")

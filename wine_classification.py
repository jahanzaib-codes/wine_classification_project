"""
DATA SCIENCE FINAL LAB EXAM – VARIANT 1
Wine Dataset Multiclass Classification with PCA & Model Deployment
Author: Jahanzaib Channa
"""

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("DATA SCIENCE FINAL LAB EXAM – VARIANT 1")
print("Wine Dataset Multiclass Classification with PCA & Model Deployment")
print("=" * 80)

# =============================================================================
# SECTION A: PRACTICAL TASKS
# =============================================================================

# =============================================================================
# Task (a): Data Loading, Cleaning & Exploration (2 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("TASK A: DATA LOADING, CLEANING & EXPLORATION")
print("=" * 80)

# 1. Load the Wine dataset
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names
target_names = wine.target_names

# 2. Display shapes of X and y
print("\n📊 Dataset Shapes:")
print(f"   X (Features) shape: {X.shape}")
print(f"   y (Target) shape: {y.shape}")
print(f"   Number of samples: {X.shape[0]}")
print(f"   Number of features: {X.shape[1]}")

# 3. Convert the dataset into a Pandas DataFrame
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
df['wine_class'] = df['target'].map({i: target_names[i] for i in range(len(target_names))})

# Show first 5 rows
print("\n📋 First 5 rows of the dataset:")
print(df.head().to_string())

# Display summary statistics
print("\n📈 Summary Statistics:")
print(df.describe().to_string())

# 4. Display class distribution using value_counts()
print("\n🏷️ Class Distribution:")
class_dist = pd.Series(y).value_counts().sort_index()
for i, count in enumerate(class_dist):
    print(f"   Class {i} ({target_names[i]}): {count} samples ({count/len(y)*100:.1f}%)")

# Determine if the dataset is balanced
min_count = class_dist.min()
max_count = class_dist.max()
balance_ratio = min_count / max_count

print(f"\n⚖️ Balance Analysis:")
print(f"   Min class count: {min_count}")
print(f"   Max class count: {max_count}")
print(f"   Balance ratio (min/max): {balance_ratio:.4f}")

if balance_ratio >= 0.8:
    print("   ✅ Dataset is BALANCED (ratio >= 0.8)")
else:
    print("   ⚠️ Dataset is IMBALANCED (ratio < 0.8)")

# =============================================================================
# Task (b): Preprocessing, Scaling & Stratified Split (2 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("TASK B: PREPROCESSING, SCALING & STRATIFIED SPLIT")
print("=" * 80)

# 1. Standardize all features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n🔧 Feature Standardization:")
print("   - Applied StandardScaler to all features")
print(f"   - Mean of scaled features: {np.mean(X_scaled):.10f} (≈ 0)")
print(f"   - Std of scaled features: {np.std(X_scaled):.4f} (≈ 1)")

# 2. Split the dataset into 80% training and 20% testing using stratified sampling
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print("\n📂 Stratified Train-Test Split:")
print(f"   Training set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(y)*100:.0f}%)")
print(f"   Testing set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(y)*100:.0f}%)")

print("\n   Training set class distribution:")
train_counts = pd.Series(y_train).value_counts().sort_index()
for i, count in enumerate(train_counts):
    print(f"      Class {i}: {count} samples ({count/len(y_train)*100:.1f}%)")

print("\n   Testing set class distribution:")
test_counts = pd.Series(y_test).value_counts().sort_index()
for i, count in enumerate(test_counts):
    print(f"      Class {i}: {count} samples ({count/len(y_test)*100:.1f}%)")

# Save scaler for deployment
joblib.dump(scaler, 'scaler.pkl')
print("\n   ✅ Scaler saved as 'scaler.pkl'")

# =============================================================================
# Task (c): PCA Analysis (3 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("TASK C: PCA ANALYSIS")
print("=" * 80)

# 1. Apply PCA and determine components needed for 95% and 99% variance
pca_full = PCA()
pca_full.fit(X_train)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance_ratio)

# Components needed for 95% variance
components_95 = np.argmax(cumulative_variance >= 0.95) + 1
# Components needed for 99% variance
components_99 = np.argmax(cumulative_variance >= 0.99) + 1

print("\n🔬 PCA Variance Analysis:")
print(f"\n   Individual Explained Variance Ratios:")
for i, var in enumerate(explained_variance_ratio):
    print(f"      PC{i+1}: {var:.6f} ({var*100:.2f}%)")

print(f"\n   Cumulative Explained Variance:")
for i, cum_var in enumerate(cumulative_variance):
    marker = ""
    if i + 1 == components_95:
        marker = " ← 95% threshold"
    elif i + 1 == components_99:
        marker = " ← 99% threshold"
    print(f"      PC{i+1}: {cum_var:.6f} ({cum_var*100:.2f}%){marker}")

print(f"\n📊 Components Needed:")
print(f"   For 95% variance: {components_95} components")
print(f"   For 99% variance: {components_99} components")

# 2. Transform training and testing data using 95% variance PCA
pca_95 = PCA(n_components=components_95)
X_train_pca = pca_95.fit_transform(X_train)
X_test_pca = pca_95.transform(X_test)

print(f"\n🔄 Data Transformation with {components_95}-component PCA (95% variance):")
print(f"   Original training shape: {X_train.shape}")
print(f"   Transformed training shape: {X_train_pca.shape}")
print(f"   Original testing shape: {X_test.shape}")
print(f"   Transformed testing shape: {X_test_pca.shape}")
print(f"   Dimensionality reduction: {X_train.shape[1]} → {X_train_pca.shape[1]} features")

# 3. Display explained variance values numerically
print(f"\n📈 Explained Variance for {components_95}-component PCA:")
for i, var in enumerate(pca_95.explained_variance_ratio_):
    print(f"   PC{i+1}: Variance = {pca_95.explained_variance_[i]:.4f}, Ratio = {var:.6f} ({var*100:.2f}%)")
print(f"\n   Total Variance Explained: {sum(pca_95.explained_variance_ratio_)*100:.2f}%")

# Save PCA for deployment
joblib.dump(pca_95, 'pca_model.pkl')
print("\n   ✅ PCA model saved as 'pca_model.pkl'")

# =============================================================================
# Task (d): Model Training, Evaluation & Comparison (3 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("TASK D: MODEL TRAINING, EVALUATION & COMPARISON")
print("=" * 80)

# Define classifiers
classifiers = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Support Vector Machine (SVM)': SVC(kernel='rbf', C=1.0, random_state=42)
}

results = {}

for name, clf in classifiers.items():
    print(f"\n{'─' * 60}")
    print(f"🤖 Training: {name}")
    print(f"{'─' * 60}")
    
    # Train the model
    clf.fit(X_train_pca, y_train)
    
    # Make predictions
    y_pred = clf.predict(X_test_pca)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = {'model': clf, 'accuracy': accuracy, 'predictions': y_pred}
    
    # 1. Report test accuracy
    print(f"\n   📊 Test Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 2. Display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   📋 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Class 0  Class 1  Class 2")
    for i in range(3):
        print(f"   Actual {i}:    {cm[i][0]:5d}    {cm[i][1]:5d}    {cm[i][2]:5d}")
    
    # Classification report
    print(f"\n   📈 Classification Report:")
    report = classification_report(y_test, y_pred, target_names=target_names, output_dict=True)
    print(f"              Precision  Recall  F1-Score  Support")
    for class_name in target_names:
        r = report[class_name]
        print(f"   {class_name:10s}    {r['precision']:.2f}     {r['recall']:.2f}     {r['f1-score']:.2f}      {int(r['support'])}")
    print(f"   {'─' * 50}")
    print(f"   Accuracy:                         {report['accuracy']:.2f}      {len(y_test)}")

# 3. Identify the best-performing classifier
print("\n" + "=" * 60)
print("🏆 MODEL COMPARISON & BEST CLASSIFIER")
print("=" * 60)

print("\n   Accuracy Summary:")
print("   " + "─" * 45)
for name, data in results.items():
    bar = "█" * int(data['accuracy'] * 40)
    print(f"   {name:35s} {data['accuracy']:.4f} {bar}")

best_model_name = max(results, key=lambda x: results[x]['accuracy'])
best_accuracy = results[best_model_name]['accuracy']
best_model = results[best_model_name]['model']

print("\n   " + "─" * 45)
print(f"\n   🥇 Best Classifier: {best_model_name}")
print(f"   📊 Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Justification
if best_model_name == 'Random Forest':
    justification = "Random Forest is the best classifier because it achieves the highest accuracy by combining multiple decision trees to reduce overfitting and improve generalization."
elif best_model_name == 'Decision Tree':
    justification = "Decision Tree is the best classifier because it achieved the highest accuracy while maintaining interpretability and fast prediction times."
else:
    justification = "SVM is the best classifier because it achieves the highest accuracy by finding the optimal hyperplane that maximizes the margin between classes."

print(f"\n   📝 Justification: {justification}")

# =============================================================================
# Task (e): Model Deployment Preparation (5 Marks)
# =============================================================================
print("\n" + "=" * 80)
print("TASK E: MODEL DEPLOYMENT PREPARATION")
print("=" * 80)

# 1. Save the best-trained model using joblib
model_filename = 'best_wine_model.pkl'
joblib.dump(best_model, model_filename)

print(f"\n💾 Model Saving:")
print(f"   ✅ Best model ({best_model_name}) saved as '{model_filename}'")
print(f"   ✅ Scaler saved as 'scaler.pkl'")
print(f"   ✅ PCA model saved as 'pca_model.pkl'")

# Save additional metadata
metadata = {
    'model_name': best_model_name,
    'accuracy': best_accuracy,
    'feature_names': list(feature_names),
    'target_names': list(target_names),
    'n_pca_components': components_95
}

joblib.dump(metadata, 'model_metadata.pkl')
print(f"   ✅ Model metadata saved as 'model_metadata.pkl'")

print("\n📱 Streamlit App:")
print("   To run the Streamlit app, execute:")
print("   streamlit run streamlit_app.py")

print("\n" + "=" * 80)
print("✅ ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 80)

# =============================================================================
# VISUALIZATION (BONUS)
# =============================================================================
print("\n📊 Generating visualizations...")

# Create a figure with multiple subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. PCA Variance Plot
ax1 = axes[0, 0]
components_range = range(1, len(cumulative_variance) + 1)
ax1.bar(components_range, explained_variance_ratio, alpha=0.7, label='Individual')
ax1.step(components_range, cumulative_variance, where='mid', color='red', 
         linewidth=2, label='Cumulative')
ax1.axhline(y=0.95, color='green', linestyle='--', label='95% threshold')
ax1.axhline(y=0.99, color='orange', linestyle='--', label='99% threshold')
ax1.set_xlabel('Principal Component')
ax1.set_ylabel('Explained Variance Ratio')
ax1.set_title('PCA Explained Variance Analysis')
ax1.legend(loc='best')
ax1.set_xticks(components_range)

# 2. Class Distribution
ax2 = axes[0, 1]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax2.bar(target_names, class_dist.values, color=colors)
ax2.set_xlabel('Wine Class')
ax2.set_ylabel('Number of Samples')
ax2.set_title('Wine Dataset Class Distribution')
for bar, count in zip(bars, class_dist.values):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(count), ha='center', va='bottom', fontweight='bold')

# 3. PCA 2D Scatter Plot
ax3 = axes[1, 0]
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)
for i, (color, name) in enumerate(zip(colors, target_names)):
    mask = y == i
    ax3.scatter(X_2d[mask, 0], X_2d[mask, 1], c=color, label=name, alpha=0.7, s=50)
ax3.set_xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]*100:.1f}%)')
ax3.set_ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]*100:.1f}%)')
ax3.set_title('PCA 2D Visualization of Wine Dataset')
ax3.legend()

# 4. Model Accuracy Comparison
ax4 = axes[1, 1]
model_names = list(results.keys())
accuracies = [results[m]['accuracy'] for m in model_names]
bar_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
bars = ax4.barh(model_names, accuracies, color=bar_colors)
ax4.set_xlim(0, 1)
ax4.set_xlabel('Accuracy')
ax4.set_title('Model Accuracy Comparison')
for bar, acc in zip(bars, accuracies):
    ax4.text(acc + 0.01, bar.get_y() + bar.get_height()/2, 
             f'{acc:.4f}', va='center', fontweight='bold')

plt.tight_layout()
plt.savefig('wine_classification_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("   ✅ Visualization saved as 'wine_classification_analysis.png'")

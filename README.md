# 🍷 Wine Classification with PCA & Model Deployment

## DATA SCIENCE FINAL LAB EXAM – VARIANT 1

**Author:** Jahanzaib Channa  
**Dataset:** Wine Dataset (Multiclass Classification)  
**Marks:** 15 Marks

---

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Task Breakdown](#task-breakdown)
- [How to Run](#how-to-run)
- [Screenshots](#screenshots)
- [Technologies Used](#technologies-used)

---

## 📖 Overview

This project implements an **advanced multiclass classification pipeline** using the Wine dataset from scikit-learn. The complete solution includes:

- ✅ Data Loading, Cleaning & Exploration
- ✅ Preprocessing, Scaling & Stratified Split
- ✅ PCA Analysis (Dimensionality Reduction)
- ✅ Model Training (Decision Tree, Random Forest, SVM)
- ✅ Model Evaluation & Comparison
- ✅ Model Deployment using Streamlit

---

## 📁 Project Structure

```
wine_classification_project/
│
├── wine_classification.py    # Main Python script (Tasks a-d)
├── streamlit_app.py          # Streamlit web application (Task e)
├── README.md                 # Project documentation
│
├── best_wine_model.pkl       # Saved best model (generated)
├── scaler.pkl                # Saved StandardScaler (generated)
├── pca_model.pkl             # Saved PCA model (generated)
├── model_metadata.pkl        # Model metadata (generated)
│
└── wine_classification_analysis.png  # Visualization (generated)
```

---

## ⚙️ Installation

### Prerequisites

Make sure you have Python 3.8+ installed on your system.

### Step 1: Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### Step 2: Install Required Packages

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib streamlit
```

Or install all at once:

```bash
pip install -r requirements.txt
```

### Requirements.txt

```
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=0.24.0
joblib>=1.0.0
streamlit>=1.0.0
```

---

## 📚 Task Breakdown

### Task A: Data Loading, Cleaning & Exploration (2 Marks)

1. ✅ Load the Wine dataset
2. ✅ Display shapes of X and y
3. ✅ Convert to Pandas DataFrame and show:
   - First 5 rows
   - Summary statistics
4. ✅ Display class distribution using `value_counts()`
5. ✅ Determine if dataset is balanced

### Task B: Preprocessing, Scaling & Stratified Split (2 Marks)

1. ✅ Standardize all features using `StandardScaler`
2. ✅ Split dataset into 80% training and 20% testing
3. ✅ Use stratified sampling

### Task C: PCA Analysis (3 Marks)

1. ✅ Apply PCA and determine:
   - Components needed for 95% variance
   - Components needed for 99% variance
2. ✅ Transform training and testing data using 95% variance PCA
3. ✅ Display explained variance values numerically

### Task D: Model Training, Evaluation & Comparison (3 Marks)

Train the following classifiers:
- ✅ Decision Tree
- ✅ Random Forest Classifier
- ✅ Support Vector Machine (SVM)

For each model:
1. ✅ Report test accuracy
2. ✅ Display confusion matrix
3. ✅ Identify best-performing classifier with justification

### Task E: Model Deployment using Streamlit (5 Marks)

1. ✅ Save the best-trained model using joblib
2. ✅ Create Streamlit application that:
   - Accepts wine chemical properties as input
   - Loads the saved model
   - Predicts the wine class (0, 1, or 2)
3. ✅ Display prediction result clearly in the app

---

## 🚀 How to Run

### Step 1: Train Models and Generate Analysis

```bash
cd wine_classification_project
python wine_classification.py
```

This will:
- Train all models
- Display analysis in console
- Save the best model (`best_wine_model.pkl`)
- Save the scaler (`scaler.pkl`)
- Save the PCA model (`pca_model.pkl`)
- Generate visualization (`wine_classification_analysis.png`)

### Step 2: Run Streamlit Application

```bash
streamlit run streamlit_app.py
```

This will open the web application in your default browser at `http://localhost:8501`

---

## 📊 Expected Output

### Console Output (wine_classification.py)

```
================================================================================
DATA SCIENCE FINAL LAB EXAM – VARIANT 1
Wine Dataset Multiclass Classification with PCA & Model Deployment
================================================================================

================================================================================
TASK A: DATA LOADING, CLEANING & EXPLORATION
================================================================================

📊 Dataset Shapes:
   X (Features) shape: (178, 13)
   y (Target) shape: (178,)
   Number of samples: 178
   Number of features: 13

🏷️ Class Distribution:
   Class 0 (class_0): 59 samples (33.1%)
   Class 1 (class_1): 71 samples (39.9%)
   Class 2 (class_2): 48 samples (27.0%)

⚖️ Balance Analysis:
   ✅ Dataset is BALANCED (ratio >= 0.8)

... (more output)

🏆 MODEL COMPARISON & BEST CLASSIFIER

   🥇 Best Classifier: Random Forest
   📊 Accuracy: 0.9722 (97.22%)
```

---

## 🖼️ Screenshot Preview

### Streamlit Application Features:

- 🎨 **Premium Dark Theme UI**
- 📊 **Interactive Sliders** for all 13 wine features
- 🔮 **Real-time Predictions** with class probabilities
- 📋 **Model Information** sidebar
- 🧪 **Quick Test** with sample data
- 📱 **Responsive Design**

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| Python 3.x | Programming Language |
| NumPy | Numerical Computing |
| Pandas | Data Manipulation |
| Matplotlib | Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Joblib | Model Serialization |
| Streamlit | Web App Framework |

---

## 📈 Model Performance Summary

| Model | Accuracy |
|-------|----------|
| Decision Tree | ~94.44% |
| Random Forest | ~97.22% |
| SVM | ~97.22% |

> **Note:** Actual results may vary slightly due to random state.

---

## 📝 Wine Dataset Features

The Wine dataset contains 13 chemical properties:

| # | Feature | Description |
|---|---------|-------------|
| 1 | Alcohol | Alcohol content (%) |
| 2 | Malic Acid | Malic acid concentration |
| 3 | Ash | Ash content |
| 4 | Alcalinity of Ash | Alcalinity of ash |
| 5 | Magnesium | Magnesium content |
| 6 | Total Phenols | Total phenolic compounds |
| 7 | Flavanoids | Flavanoid content |
| 8 | Nonflavanoid Phenols | Non-flavanoid phenols |
| 9 | Proanthocyanins | Proanthocyanin content |
| 10 | Color Intensity | Color intensity |
| 11 | Hue | Hue (color ratio) |
| 12 | OD280/OD315 | OD280/OD315 ratio |
| 13 | Proline | Proline amino acid |

---

## 🎓 Author

**Jahanzaib Channa**  
Data Science Final Lab Exam – Variant 1

---

## 📄 License

This project is created for educational purposes as part of a Final Lab Exam.

---

## 🙏 Acknowledgments

- Scikit-learn for providing the Wine dataset
- Streamlit for the amazing web framework
- Course instructors for the exam guidelines

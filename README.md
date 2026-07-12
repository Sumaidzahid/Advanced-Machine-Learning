# Car Price Prediction & Data Science Analysis

A comprehensive machine learning project focused on predicting used car prices using advanced regression techniques, feature engineering, model interpretation, and unsupervised learning methods. The project demonstrates a full data science pipeline from raw data processing to model explainability and dimensionality reduction.

## Project Overview

This project explores a rich used car dataset to build predictive models for price estimation. It combines traditional regression techniques with ensemble learning, feature selection, and advanced interpretability methods. In addition, unsupervised learning techniques are applied to uncover hidden structures in the dataset.

The workflow covers data preprocessing, exploratory analysis, modeling, evaluation, and deep insight extraction using multiple machine learning paradigms.

## Key Components

### Data Preprocessing & EDA

* Loading and inspecting dataset
* Exploratory Data Analysis (EDA)
* Handling missing values
* Outlier detection and treatment
* Data cleaning and transformation

### Feature Engineering

* Advanced feature creation
* Feature scaling and transformation
* Feature selection using **SelectKBest**

### Supervised Learning Models

* Linear Regression
* Random Forest Regressor
* Gradient Boosting Regressor
* Polynomial Regression (Generalized Linear Model approach)

### Model Optimization & Ensembling

* Hyperparameter-aware model comparison
* Ensemble learning using **Voting Regressor**
* Model performance comparison and validation

### Model Evaluation

* R² Score
* Mean Absolute Error (MAE)
* Residual analysis
* Error distribution visualization

### Model Interpretability

* Permutation Feature Importance
* SHAP (SHapley Additive exPlanations) analysis

### Unsupervised Learning & Data Insights

* Dimensionality Reduction:

  * PCA (Principal Component Analysis)
  * MDS (Multidimensional Scaling)
  * Isomap
  * t-SNE
* Clustering Techniques:

  * K-Means Clustering
  * DBSCAN Clustering

## Tools & Technologies

* Python
* Pandas, NumPy
* Scikit-learn
* Matplotlib, Seaborn
* SHAP
* Machine Learning & Data Science libraries

## Project Workflow

1. Data loading and inspection
2. Data cleaning and preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature engineering and selection
5. Model training and comparison
6. Ensemble learning (Voting Regressor)
7. Model evaluation and visualization
8. Interpretability analysis (SHAP, permutation importance)
9. Unsupervised learning exploration (PCA, clustering, etc.)

## Results
- Reduced dataset size by 8.57% through IQR-based outlier removal while improving data quality.
- Feature selection was unnecessary, as the full feature set achieved slightly better performance than SelectKBest.
- Random Forest achieved the highest individual predictive performance (Test R² = 0.925, MAE ≈ £1,469) but showed mild overfitting.
- Gradient Boosting delivered stable performance with minimal overfitting (Test R² = 0.796, MAE ≈ £2,450).
- A weighted Voting Regressor outperformed Gradient Boosting and provided strong generalization (Test R² ≈ 0.90, CV MAE ≈ £1,734).
- Vehicle age, vehicle model, and mileage per year were identified as the most influential features using Permutation Importance and SHAP.
- SHAP and PDP analyses confirmed realistic vehicle depreciation trends, with increasing age and mileage reducing predicted prices.
- t-SNE produced the clearest separation between vehicle price groups compared with PCA, MDS, and Isomap.
- Polynomial regression (degree 2-3) provided the best fit, capturing non-linear depreciation without underfitting or overfitting.
- K-Means (k = 3) identified well-separated clusters, while DBSCAN detected one dominant cluster with scattered outliers.

## Key Insights

* Ensemble methods improved predictive stability compared to individual models
* Feature selection significantly reduced noise and improved performance
* SHAP and permutation importance provided clear model interpretability
* Dimensionality reduction revealed hidden structure in the dataset
* Clustering showed natural groupings of vehicle types and price ranges

## Author

Muhammad Zahid
MSc Artificial Intelligence | Mechatronics Engineer

## Association 

Univesity project: Manchester metropolitan university (mmu)

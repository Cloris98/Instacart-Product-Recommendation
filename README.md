# Instacart Reorder Prediction System

This project develops a machine learning pipeline to predict which products users are likely to reorder, using the Instacart Online Grocery Shopping Dataset. It consists of two main components: exploratory modeling and final classification pipeline implementation.

### Objectives

- Predict future product reorders per user.
- Build recommendation systems using collaborative filtering and supervised learning.
- Evaluate performance using classification metrics and ranking metrics.

### Dataset

- Sourced from Kaggle: [Instacart Market Basket Analysis](https://www.kaggle.com/c/instacart-market-basket-analysis/data)
- Includes 3+ million orders from 200,000+ users.

### Data Preprocessing

- Merged multiple CSV files to create user-product interaction features.
- Applied quality checks and constructed labels for supervised learning.
- Extracted behavior-based features (e.g., reorder ratio, order frequency).
- Encoded categorical data with one-hot encoding.

###  Modeling Approaches

##### From Notebook

- Used User-/Item-based collaborative filtering.
- Implemented matrix factorization using LightFM.
- Evaluated performance with MAP, Precision, and Recall.
- 
##### From Script

- Machine Learning Pipeline using:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - Gradient Boosting
  - Neural Network
- Applied hyperparameter tuning with `GridSearchCV`.
- Selected top 15 features (e.g., total product orders, mean reordered ratio).
- Addressed class imbalance using `imblearn` (RandomUnderSampler and RandomOverSampler).

## Results

- Gradient Boosting achieved best performance: ROC AUC ≈ 0.78, F1-score ≈ 0.38.
- LightFM outperformed baselines in collaborative filtering ranking tasks.

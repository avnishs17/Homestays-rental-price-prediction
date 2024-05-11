# Homestay Price Prediction Model

## Project Overview
This repository contains two Jupyter notebooks that detail the process of predicting homestay rental prices. The project is designed to handle preprocessing and predictive modeling using machine learning techniques. The aim is to provide insights into the factors affecting rental prices and to develop accurate price prediction models.

### Objectives
- **Data Cleaning and Preprocessing:** Prepare the dataset for modeling by handling missing data, removing duplicates, and encoding categorical variables.
- **Predictive Modeling:** Implement and optimize multiple regression models to predict the log prices of homestay rentals effectively.

## Methodologies

### Data Preprocessing
- **File:** `homestays-preprocessing.ipynb`
- **Techniques Used:**
  - Removing duplicate entries to ensure data integrity.
  - Handling missing values to improve model accuracy.
  - Encoding categorical variables like `zipcode` and `neighbourhood` through frequency encoding to reduce dimensionality and prepare data for regression analysis.

### Predictive Modeling
- **File:** `housestays-model.ipynb`
- **Models Deployed:**
  - **Linear Regression:** Serves as a baseline for performance comparison.
  - **Random Forest Regressor:** Used for its ability to handle non-linear relationships and provide feature importance metrics.
  - **Gradient Boosting Regressor:** Implemented to improve predictions by addressing biases and variances in the model.
- **Model Optimization:**
  - Hyperparameter tuning using GridSearchCV to find optimal settings.
  - Cross-validation using KFold to ensure the model's generalizability.

## Key Findings and Insights
- **Feature Importance:** Identifying significant predictors of rental prices such as location (encoded `zipcode`) and property characteristics.
- **Model Performance:** Gradient Boosting Regressor showed the best performance based on metrics like MSE and R-squared, suggesting its effectiveness in capturing complex patterns in the data.

## Recommendations
- **Model Enhancements:** Explore more sophisticated ensemble techniques like Stacked Regressors or XGBoost for potentially better performance.
- **Data Augmentation:** Incorporate additional data such as seasonal trends or economic indicators that might affect rental prices.

## Installation and Dependencies
- Python 3.x
- Libraries: Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Install the required libraries using: pip install pandas numpy matplotlib seaborn scikit-learn



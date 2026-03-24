# Boston Housing Regression Analysis in R

This project analyzes the Boston Housing dataset using multiple statistical and machine learning approaches in R. It includes exploratory data visualization, model training, and model evaluation across several regression techniques.

## Project Overview

The analysis uses the Boston dataset and compares the performance of:

- Multiple Linear Regression
- Best Subset Selection using BIC
- Lasso Regression
- Generalized Additive Model (GAM) / Smoothing Splines

The workflow includes:

- Exploratory plots
- 80/20 train-test split
- Model fitting
- Mean Squared Error (MSE) evaluation
- R-squared calculation
- Residual diagnostics

## Libraries Used

- ISLR2
- car
- glmnet
- gam
- splines
- leaps

## File Included

- `boston_housing_analysis.R`: main R script for data analysis, modeling, and plotting

## Methods Used

### 1. Exploratory Data Analysis
The project begins with:
- Scatter plot of `medv` vs `rm`
- Histogram of `rm`
- Histogram of `age`
- Boxplot of `rad` as a factor vs `nox`

### 2. Multiple Linear Regression
A full multiple linear regression model is trained using all predictors to predict `crim`.

### 3. Best Subset Selection
Exhaustive best subset selection is performed using BIC to identify the most relevant predictor combination.

### 4. Lasso Regression
Lasso regression is trained using cross-validation to identify the optimal lambda and reduce model complexity.

### 5. GAM / Smoothing Splines
A generalized additive model is built using smoothing splines for nonlinear relationships.

## Evaluation Metrics

The following metrics are used:
- Training MSE
- Test MSE
- Training R-squared
- Test R-squared
- Residual plots

## How to Run

1. Open the script in RStudio
2. Install required packages if needed:
   ```r
   install.packages(c("ISLR2", "car", "glmnet", "gam", "splines", "leaps"))

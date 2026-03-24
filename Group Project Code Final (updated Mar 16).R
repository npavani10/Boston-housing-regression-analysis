#Load Libraries
library(ISLR2)
library(car)
library(glmnet)
library(gam)
library(splines)
library(leaps)
attach(Boston)


##########################################

##################Task 1 part 1 plots##########################

# Scatter plot of medv vs. rm
plot(medv,rm)

# Histogram showing rm frequency
hist_for_counts <- hist(rm, breaks = 9, plot = FALSE)
hist(rm, breaks = 9,
     main = paste("Histogram of rm"),
     xlab = "rm",
     ylab = "Frequency",
     xlim = c(3, max(rm)),
     ylim = c(0, max(hist_for_counts$counts) * 1.1)
)
text(x = hist_for_counts$mids,
     y = hist_for_counts$counts + 5,
     labels = hist_for_counts$counts,
     cex = 0.8,
     col = "black"
)

# Histogram showing age frequency
hist_for_counts_age <- hist(age, breaks = 10, plot = FALSE)
hist(age, breaks = 9,
     main = paste("Histogram of age"),
     xlab = "age",
     ylab = "Frequency",
     xlim = c(3, max(age)),
     ylim = c(0, max(hist_for_counts_age$counts) * 1.1)
)
text(x = hist_for_counts_age$mids,
     y = hist_for_counts_age$counts + 5,
     labels = hist_for_counts_age$counts,
     cex = 0.8,
     col = "black"
)

#Box plot shoing rad as a factor vs nox
rad_factor <- as.factor(rad)
plot(rad_factor, nox,
     xlab = "rad",
     ylab = "nox")

################# Part 2 of Task 1 multiple linear reg #####################

# Creating random 80-20 split of data for training and testing

#below makes it so the random selection is the same every time
set.seed(25)
data_set <- Boston

#produces row numbers that will be in training set
train_rows <- sample(1:nrow(data_set), size = 0.8 * nrow(data_set))
#below is used in Lasso, used to make inverse of training data
test_rows <- -train_rows

# Creating the train and test split based on the 80/20 split above
# - is used in test_data so the inverse of the rows used in the
# train data set are used. The empty argument after the rows argument
# specifies that all columns are wanted

train_data <- data_set[train_rows, ]
test_data <- data_set[-train_rows, ]

#Below checks that size of each data set just to make sure the split
# worked correctly
cat("Training set size: ", nrow(train_data), "\n")
cat("Test set size: ", nrow(test_data), "\n")

#Creates multiple linear regression model using all variables
linear_reg_training_model <- lm(crim ~ ., data = train_data)
summary(linear_reg_training_model)
vif(linear_reg_training_model)

#Calc MSE of full MLR model for training data
MSE_train <- mean((predict(linear_reg_training_model, train_data) - train_data$crim)^2)
cat("Training MSE of MLR Model: ", MSE_train)

# Predicting Multiple linear reg
actual_test_data <- test_data$crim

predicted_linear_reg_test_results <- predict(linear_reg_training_model, test_data)

#Calc MSE of full MLR model for test data
MSE_test <- mean((predicted_linear_reg_test_results - actual_test_data)^2)
cat("Test MSE of MLR Model: ", MSE_test)


# Plots assess model #

#Below plots are fit for training model
plot(predict(linear_reg_training_model),residuals(linear_reg_training_model), 
     xlab = "Fitted values",
     ylab = "Residuals")
#creates a dashed line at zero
abline(h = 0, col = "black", lty = 2)
#creates a line to see if there is a trend in the data
lines(lowess(predict(linear_reg_training_model), 
             residuals(linear_reg_training_model)
             ), 
      col = "red", lwd = 2
      )


################### Part 3 Best Subset Selection ##########

# Perform Exhaustive Best Subset Selection using BIC
bic_fit <- regsubsets(crim ~ ., data = train_data, nvmax = ncol(train_data) - 1, method = "exhaustive")

# Extract model selection summary
bic_summary <- summary(bic_fit)

# Identify the best model based on BIC
best_bic_model <- which.min(bic_summary$bic)
best_bic_model

# Extract the predictors selected in the best BIC model
selected_predictors <- coef(bic_fit, id = best_bic_model)
selected_predictors
cat("Selected Predictors for BIC Model:\n")
print(names(selected_predictors))

# Define a custom predict function for regsubsets objects
predict.regsubsets <- function(object, newdata, id, ...) {
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form, newdata)
  coefs <- coef(object, id = id)
  xvars <- names(coefs)
  mat[, xvars] %*% coefs
}

# Function to compute MSE and R²
compute_metrics <- function(model, data, response_var, id) {
  preds <- predict.regsubsets(model, newdata = data, id = id)
  mse <- mean((data[[response_var]] - preds)^2)
  ss_total <- sum((data[[response_var]] - mean(data[[response_var]]))^2)
  ss_residual <- sum((data[[response_var]] - preds)^2)
  r_squared <- 1 - (ss_residual / ss_total)
  residuals <- data[[response_var]] - preds
  return(list(MSE = mse, R2 = r_squared, Residuals = residuals, Predictions = preds))
}

# Compute metrics for Train & Test sets using the BIC-selected model
train_metrics <- compute_metrics(bic_fit, train_data, "crim", best_bic_model)
test_metrics <- compute_metrics(bic_fit, test_data, "crim", best_bic_model)

# Print model performance results
cat("\n### Model Performance for BIC-Selected Model ###\n")
cat("Train MSE:", train_metrics$MSE, "\n")
cat("Train R²:", train_metrics$R2, "\n")
cat("Test MSE:", test_metrics$MSE, "\n")
cat("Test R²:", test_metrics$R2, "\n")

# Residual plot for Train Set
plot(train_metrics$Predictions, train_metrics$Residuals,
     main = "Residual Plot (Train Set)", xlab = "Fitted Values", ylab = "Residuals",
     col = "blue", pch = 19)
abline(h = 0, col = "red", lwd = 2)


########### Part 4 Lasso ############

View(data_set)

#creates an x matrix with the predictors only
# [, -1] is used so the first column (crim) is not used in the 
# x matrix
x <- model.matrix(crim ~., data_set)[, -1]

#creates a vector with all of the respondent variable values
y <- data_set$crim

# creates a grid ranging from lambda = 10^10 to lambda = 10^-2
grid <- 10^seq(10, -2, length = 100)

#creates model using lasso based on the training data
lasso_training_model <- glmnet(x[train_rows, ], y[train_rows], 
                               alpha = 1, lambda = grid)
plot(lasso_training_model)


#performs cross validation on the training data set to see which
#lambda has the smallest MSE
cv_lasso <- cv.glmnet(x[train_rows, ], y[train_rows], alpha = 1)
plot(cv_lasso)

#stores the lambda that gives lowest MSE in a variable called bestlambda
bestlambda <- cv_lasso$lambda.min
bestlambda

#checks to see how well the model fits the test set of data, creates 
# predicted values of the test set based on model
pred_lasso <- predict(lasso_training_model, s = bestlambda, newx = x[test_rows, ])

# uses the x matrix to predict the y vector defined above.
# alpha = 1 indicates this should be lasso regression
# lambda = grid indicates the grid of lambdas defined above should be 
# used to creat many models that all get stored in out
out <- glmnet(x, y, alpha = 1, lambda = grid)

# extracts coefficients from from lasso model that corresponds 
# to the bestlambda
# the [1:13, ] is used to make sure all of the predictor 
# coefficients are outputted.
lasso_coef <- predict(out , type = "coefficients",
                          s = bestlambda)[1:13 , ]

lasso_coef

# Showing prediction performance of Lasso##

predict_lasso_training_model <- predict(lasso_training_model, 
                                        s = bestlambda, 
                                        newx = x[train_rows, ]
)

#Calculates MSE of the training set
MSE_lasso_train <- mean((predict_lasso_training_model - y[train_rows])^2)
cat("Train MSE of Lasso: ", MSE_lasso_train)

#Calculates MSE of the test set
MSE_lasso_test <- mean((pred_lasso - y[test_rows])^2)
cat("Test MSE of Lasso: ", MSE_lasso_test)

#Calc R^2 of training set on Lasso
TSS_lasso_train <- sum((y[train_rows] - mean(y[train_rows]))^2)
RSS_lasso_train <- sum((y[train_rows] - predict_lasso_training_model)^2)
r_squared_lasso_train <- 1 - (RSS_lasso_train / TSS_lasso_train)
cat("Training R^2 for Lasso: ", r_squared_lasso_train)

#residual plot for Lasso training data
plot(predict_lasso_training_model, residuals(lasso_training_model, s = bestlambda), 
     xlab = "Fitted values",
     ylab = "Residuals")

#creates a dashed line at zero
abline(h = 0, col = "black", lty = 2)

#creates a line to see if there is a trend in the data
lines(lowess(predict_lasso_training_model, 
             residuals(lasso_training_model, s = bestlambda)
            ), 
      col = "red", lwd = 2
      )



########## Part 5 Smoothing spline ##########

# Creates GAM model for smoothing spline with arbitrary degrees of freedom
smoothing_gam_model_train <- gam(crim ~ s(zn, 5) + s(indus, 5) + s(nox, 5) + 
                   s(rm, 5) + s(age, 5) + s(dis, 5) + 
                   s(rad, 5) + s(tax, 5) + s(ptratio, 5) + 
                   s(lstat, 5) + s(medv, 5), 
                 data = train_data
                 )

summary(smoothing_gam_model_train)

# Spline prediction on test data
predicted_spline_test_results <- predict(smoothing_gam_model_train, newdata = test_data)

# Calculates MSE for train data
MSE_spline_train <- mean((predict(smoothing_gam_model_train, train_data)-train_data$crim)^2)
cat("MSE of test set:", MSE_spline_train, "\n")

# Calc R^2 for spline training data
TSS_spline_train <- sum((y[train_rows] - mean(y[train_rows]))^2)
RSS_spline_train <- sum((y[train_rows] - predict(smoothing_gam_model_train, train_data))^2)
r_squared_spline_trian <- 1 - (RSS_spline_train/ TSS_spline_train)
cat("Train R^2 for Spline: ", r_squared_spline_trian)

# Calculates MSE for test data
MSE_spline_test <- mean((actual_test_data - predicted_spline_test_results)^2)
cat("MSE of test set:", MSE_spline_test, "\n")

#below used to calc residuals for the actual test data against the 
#predicted training data
spline_test_data_residuals <- actual_test_data - predicted_spline_test_results

plot(predicted_spline_test_results ,spline_test_data_residuals, 
     xlab = "Fitted values",
     ylab = "Residuals")

#creates a dashed line at zero
abline(h = 0, col = "black", lty = 2)
#creates a line to see if there is a trend in the data
lines(lowess(predict(smoothing_gam_model_train), 
             residuals(smoothing_gam_model_train)
            ), 
      col = "red", lwd = 2
      )


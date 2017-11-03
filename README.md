# sys6018-competition-safe-driver

# Team Roles

##Ni: Data Cleaning, Linear Model, and Random Forest Models
##Morgan: Cross Validation
##Boh: Refining the models
##All:  Reflection Paper

# Model Selection

##We chose our linear model based on using a stepwise process on a sample of the training data in which variables were added to or taken out of the model in the order that minimized MSE. After determining the model that performed the best on a sample of the training data using this method, we used K fold cross validation to validate our results.  Additionally we determined the model that maximized the normalized gini index on the training sample and then validated this result by calculating the normalized gini index for the entire training data set.  

##Our kaggle score for the final linear momdel submitted was 0.156

##We determined the random forest model to use by testing various values of m for the model and determining the model with the lowest OOB error rate.  we then validated this by calculating the error rate on a testing data set and additionally calculated the normalized gini index for the entire training dataset.  

##The kaggle score for the Random Forest model submitted was also 0.156


train = read.csv("C:\\Users\\mkw5c\\OneDrive\\Documents\\SYS6018\\competitions\\train.csv")
test = read.csv("C:\\Users\\mkw5c\\OneDrive\\Documents\\SYS6018\\competitions\\test.csv")

library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('rlang') # data manipulation
library(randomForest)
# -----Data Exploration------- #
summary(train)
summary(test)

# drop NA rows
newdata <- na.omit(train) # no NA rows
newdata_test <- na.omit(test) # no NA rows

# remove duplicates
train = newdata[!duplicated(newdata), ] # no duplicates
test = newdata_test[!duplicated(newdata_test), ] # no duplicates

# fill the missing value -1 with mode using which.max(x)
for (i in 2:59){
  train[,i][train[,i]=='-1'] <- which.max(train[,i])
}


# turn variables ending with cat to factor and bin to logical 
train[,grep("cat", names(train))] <-lapply(train[,grep("cat", names(train))], as.factor)
train[,grep("bin", names(train))] <-lapply(train[,grep("bin", names(train))], as.logical)
str(train) # check to see the type of each variable

test[,grep("cat", names(test))] <-lapply(test[,grep("cat", names(test))], as.factor)
test[,grep("bin", names(test))] <-lapply(test[,grep("bin", names(test))], as.logical)
str(test)

train_sample = train[sample(nrow(train), 3000), ]



apply(train_sample,2,unique)

# --------- Linear Model --------# 
# Linear model (free to use any packaged implementations). Use only the supplied training data.
# Linear model (free to use any packaged implementations). Use only the supplied training data.
training<- sample(595212, 10000, replace = FALSE)
train.train = train[training, ]
train.test = train[-training, ]

fit = glm(target~., data = train_sample)
summary(fit)
anova(fit)

# stepwise model selection 
start<-glm(target ~1,family = binomial(link = "logit"), data= train.train)
end<-glm(target~., binomial(link = "logit"), data= train.train)
result.s<-step(start, scope=list(upper=end), direction="both",trace=FALSE) 
summary(result.s)
anova(result.s)



#see how result.s performs on the full dataset
driver.lm<- glm(target ~ ps_reg_03 + ps_car_13 + ps_calc_08 + ps_ind_15 + 
                  ps_car_11 +  ps_calc_09 + ps_ind_06_bin, family = binomial(link = "logit"), 
                data = train)
pred_probs<- predict(driver.lm, newdata = test, type = "response")

normalized.gini.index(ground.truth = train$target, predicted.probabilities = pred_probs)


cv_error<- cv.glm(train, driver.lm, K = 5)
cv_error$delta

##training error = 3.4%

###Gini Index
unnormalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  if (length(ground.truth) !=  length(predicted.probabilities))
  {
    stop("Actual and Predicted need to be equal lengths!")
  }

  # arrange data into table with columns of index, predicted values, and actual values
  gini.table = data.frame(index = c(1:length(ground.truth)), predicted.probabilities, ground.truth)
  
  # sort rows in decreasing order of the predicted values, breaking ties according to the index
  gini.table = gini.table[order(-gini.table$predicted.probabilities, gini.table$index), ]
  
  # get the per-row increment for positives accumulated by the model 
  num.ground.truth.positivies = sum(gini.table$ground.truth)
  model.percentage.positives.accumulated = gini.table$ground.truth / num.ground.truth.positivies
  
  # get the per-row increment for positives accumulated by a random guess
  random.guess.percentage.positives.accumulated = 1 / nrow(gini.table)
  
  # calculate gini index
  gini.sum = cumsum(model.percentage.positives.accumulated - random.guess.percentage.positives.accumulated)
  gini.index = sum(gini.sum) / nrow(gini.table) 
  return(gini.index) }

unnormalized.gini.index(ground.truth = train$target, predicted.probabilities = fitted(driver.lm))

##unnormalized gini index = 0.098

normalized.gini.index = function(ground.truth, predicted.probabilities) {
  
  model.gini.index = unnormalized.gini.index(ground.truth, predicted.probabilities)
  optimal.gini.index = unnormalized.gini.index(ground.truth, ground.truth)
  return(model.gini.index / optimal.gini.index)}

normalized.gini.index(ground.truth = train$target, predicted.probabilities = fitted(driver.lm))
 
##Normalized Gini Index = 0.203

# ----------Random Forest-----------# 
# Random Forest 
set.seed(1)
tr <- sample(1:nrow(train_sample), nrow(train_sample) / 2)
train_sample.train <- train_sample[tr, ]
train_sample.test <- train_sample[-tr, ]

rf.train_sample <- randomForest(as.factor(target) ~ ps_reg_03 + ps_car_13 + ps_calc_08 + ps_ind_15 + 
                                  ps_car_11 +  ps_calc_09 + ps_ind_06_bin, data = train_sample, mtry = 6, ntree = 500, importance = TRUE)

rf.train_sample # OOB estimate of error rate is 4% 

rf.train_sample <- randomForest(as.factor(target) ~.-ps_car_11_cat, data = train_sample, mtry = 6, ntree = 500, importance = TRUE)
rf.train_sample # 4% 

yhat.rf.train<- predict(rf.train_sample, newdata = train, type = "prob")
normalized.gini.index(ground.truth = train$target, predicted.probabilities = yhat.rf.train)

test_predictions_rf <- predict(rf.train_sample, newdata = test, type = "prob")
test_predictions_rf[,2]
test_sub<- cbind(test$id, test_predictions_rf[,2])
mode(test$id)
write.table(test_sub, file = "safe_driver_rf.csv", sep = ",", row.names = F, col.names = c("id", "target"))

yhat.rf<- predict(rf.train_sample, newdata = train, type = "response")
table(yhat.rf, train$target)
(573383+121)/595212
# correction rate: 0.9613333

yhat.rf2 <- predict(rf.train_sample, newdata = train_sample.test)
yhat.rf2<- as.numeric(yhat.rf2)
normalized.gini.index(ground.truth = train_sample.test$target, predicted.probabilities = yhat.rf)

importance(rf.train_sample)


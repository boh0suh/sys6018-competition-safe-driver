setwd("/Users/nizhuang/Documents/Data Science /Sys Competition /sys6018-competition-safe-driver-master")
train = read.csv("train.csv")
test = read.csv('test.csv')

library('dplyr') # data manipulation
library('readr') # input/output
library('data.table') # data manipulation
library('tibble') # data wrangling
library('tidyr') # data wrangling
library('stringr') # string manipulation
library('forcats') # factor manipulation
library('rlang') # data manipulation
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
fit = glm(target~.,family=binomial(link='logit'), data = train_sample)
summary(fit)
anova(fit)

# stepwise model selection 
start<-glm(target ~1,family=binomial(link='logit'),data= train_sample)
end<-glm(target~.,family=binomial(link='logit'),data= train_sample)
result.s<-step(start, scope=list(upper=end), direction="both",trace=FALSE) 
summary(result.s)
anova(result.s)

predict(result.s, newdata = train_sample.test, type = 'response')

#k-fold cross validation for logistic regression 
#https://www.r-bloggers.com/predicting-creditability-using-logistic-regression-in-r-cross-validating-the-classifier-part-2-2/
library(boot)
cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)
cv.glm(train_sample,result.s,K=5,cost=cost)$delta[1]
# 0.03633333


# ----------Random Forest-----------# 
# Random Forest 
set.seed(1)
tr <- sample(1:nrow(train_sample), nrow(train_sample) / 2)
train_sample.train <- train_sample[tr, ]
train_sample.test <- train_sample[-tr, ]

rf.train_sample <- randomForest(as.factor(target) ~.-ps_car_11_cat, data = train_sample.train, mtry = 30, ntree = 500, importance = TRUE)
rf.train_sample # OOB estimmate of error rate is 4% 

rf.train_sample <- randomForest(as.factor(target) ~.-ps_car_11_cat, data = train_sample.train, mtry = 6, ntree = 500, importance = TRUE)
rf.train_sample # 4% 

yhat.rf <- predict(rf.train_sample, newdata = train_sample.test)
yhat.rf
table(yhat.rf,train_sample.test$target)
(1442+0)/1500
# correction rate: 0.9613333

importance(rf.train_sample)


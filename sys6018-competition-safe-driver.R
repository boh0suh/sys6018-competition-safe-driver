setwd("/Users/nizhuang/Documents/Data Science /Sys Competition /sys6018-competition-safe-driver-master")
train = read.csv("train.csv")
test = read.csv('test.csv')

library('dplyr') # data manipulation
library('readr') # input/output
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

# fill the missing value -1 with mode using which.max(x) for train 
for (i in 2:59){
  train[,i][train[,i]=='-1'] <- which.max(train[,i])
}


dim(test)
# fill the missing value -1 with mode using which.max(x) for test 
for (i in 2:58){
  test[,i][test[,i]=='-1'] <- which.max(test[,i])
}



# turn variables ending with cat to factor and bin to logical 
train[,grep("cat", names(train))] <-lapply(train[,grep("cat", names(train))], as.factor)
train[,grep("bin", names(train))] <-lapply(train[,grep("bin", names(train))], as.logical)
str(train) # check to see the type of each variable

test[,grep("cat", names(test))] <-lapply(test[,grep("cat", names(test))], as.factor)
test[,grep("bin", names(test))] <-lapply(test[,grep("bin", names(test))], as.logical)
str(test)


# because variables have different levels in train and test data 
train$ps_ind_05_cat<- NULL
test$ps_ind_05_cat<-NULL
train$ps_car_05_cat<-NULL
test$ps_car_05_cat<-NULL
train$ps_ind_02_cat<-NULL
test$ps_ind_02_cat<-NULL
train$ps_car_03_cat<- NULL
test$ps_car_03_cat<- NULL

common <- intersect(names(train[,grep("cat", names(train))]), names(test[,grep("cat", names(test))])) 
for (p in common) { 
  levels(test[[p]]) <- levels(train[[p]]) 
}

set.seed(1)
train_sample = train[sample(nrow(train), 10000), ]



# --------- Linear Model --------# 
# Linear model (free to use any packaged implementations). Use only the supplied training data.
fit = glm(target~., data = train_sample)
summary(fit)
anova(fit)

# stepwise model selection 
start<-glm(target ~1,family = binomial(link = "logit"),data= train_sample)
end<-glm(target~.,family = binomial(link = "logit"),data= train_sample)
result.s<-step(start, scope=list(upper=end), direction="both",trace=FALSE) 
summary(result.s)
anova(result.s)


#k-fold cross validation for logistic regression 
#https://www.r-bloggers.com/predicting-creditability-using-logistic-regression-in-r-cross-validating-the-classifier-part-2-2/
library(boot)
cost <- function(r, pi = 0) mean(abs(r-pi) > 0.5)
cv.glm(train_sample,result.s,K=5,cost=cost)$delta[1]
# 0.03733333


pred <-predict(result.s, newdata = test, type = 'response')
pred <- ifelse(pred> 0.5, 1, 0)
sum(pred!=0)


table = data.frame(test$id,pred)
write.table(table,file="safedriver_lm.csv",sep = ',', row.names = F,col.names = c('id','target'))


# ----------Random Forest-----------# 

# Random Forest 
set.seed(1)
tr <- sample(1:nrow(train_sample), nrow(train_sample) / 2)
train_sample.train <- train_sample[tr, ]
train_sample.test <- train_sample[-tr, ]

rf.train_sample <- randomForest(as.factor(target) ~.-ps_car_11_cat, data = train_sample.train, mtry = 30, ntree = 500, importance = TRUE)
rf.train_sample # OOB estimmate of error rate is 3.44% 

rf.train_sample <- randomForest(as.factor(target) ~.-ps_car_11_cat, data = train_sample.train, mtry = 6, ntree = 500, importance = TRUE)
rf.train_sample # 3.4% 

yhat.rf <- predict(rf.train_sample, newdata = train_sample.test,type = 'response')
yhat.rf

table(yhat.rf,train_sample.test$target)
(4797)/5000
# correction rate:  0.9594
importance(rf.train_sample)

pred <-predict(rf.train_sample, newdata = test, type = 'prob')
sum(is.na(pred))
table = data.frame(test$id,pred[,2])
write.table(table,file="safedriver_rf.csv",sep = ',', row.names = F,col.names = c('id','target'))


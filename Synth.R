# Case Study Solutions : synth.tr.csv and synth.te.csv file
#---------------------------------------------------------------------------------------------------------------

library(ROCR) #Creating ROC curve
library(PRROC) #Precision-recall curve
library(glmnet) #Lasso
library(tidyverse)
library(DT)
library(glmnet)
library(rpart)
library(rpart.plot)
library(caret)
library(knitr)
library(mgcv)
library(nnet)
library(NeuralNetTools)
library(knitr)
library(dplyr)
library(tidyr)
library(reshape2)
library(RColorBrewer)
library(GGally)
library(ggplot2)
library(caret)
library(glmnet)
library(boot)
library(verification)

#---------------------------------------------------------------------------------------------------------------

# Soln. to Question 1:

# Reading the CSV file and dropping the first column
data=read.csv('synth.tr.csv')
data_test=read.csv('synth.te.csv')

# View the train data loaded
data
# Dropping the first column which is nothing but the Serial number
data=data[2:4]
data_test=data_test[2:4]
# View the dimensions (shape) of the data to be used for the analysis
dim(data)
dim(data_test)

data$yc<-as.factor(data$yc)
data_test$yc<-as.factor(data_test$yc)
#---------------------------------------------------------------------------------------------------------------

# Soln. to Question 2:

# The key objective of the case study is to understand how the classification works. It's a typical 2 class 2 feature problem.
# Both the train and test dataset have two features : X and Y with 2 classes : 1 and 0 for yc (response variable)
# We plan to construct a decision rule or run any of the classification algorithms to classify the sample points in the testing data to the respective categories

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 3:

library(ggplot2)
ggplot(data) + geom_point(aes(x = xs, y = ys, colour = factor(yc)))

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 4:

dup_check_xs=dim(data[duplicated(data$xs),])[1]
dup_check_ys=dim(data[duplicated(data$ys),])[1]

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 5:

dup_check_xs
dup_check_ys

# Both xs and ys features in the given training dataset has no duplicates

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 6:

par(mfrow=c(1,1)) # set the plotting into a 1*2 array

ggplot(data) + geom_point(aes(x = xs, y = ys, colour = factor(yc)))
ggplot(data_test) + geom_point(aes(x = xs, y = ys, colour = factor(yc)))


#---------------------------------------------------------------------------------------------------------------
# Solution to Question 7:

hist(data$xs, col='blue', xlim=c(-1.5, 1))
hist(data$ys, col='red',xlim=c(-0.5,1))

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 8:

# Supervised machine learning: because we have labeled data here in the given dataset. 
# Here: the features (X and Y) are the lables : independent variables : xs and ys for the response/target variable : yc
# We can plan to construct a decision rule bsaed on which we can classify the sample points for the respective categories

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 14:

trainrows <- sample(nrow(data), nrow(data) * 0.70)
data.train <- data[trainrows, ]
data.test <- data[-trainrows,]

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 10:

data.train.glm0 <- glm(yc~., family = binomial, data.train)
summary(data.train.glm0)

# We observe an AIC score of 355.51 with residual deviance of 349.51 on 697 degrees of freedom
#---------------------------------------------------------------------------------------------------------------
# Solution to Question 11:

pred <- prediction(data.train.pred, data.train$yc)
perf <- performance(pred, "tpr", "fpr")
plot(perf, colorize=TRUE)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 12:

#Get the AUC
unlist(slot(performance(pred, "auc"), "y.values"))

# Logistic Regression : 95%

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 13:

library(randomForest)

m3 <- randomForest(yc ~ ., data = data.train)

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 14:

summary(m3)

m3_fitForest <- predict(m3, newdata = data.test, type="prob")[,2]

m3_pred <- prediction( m3_fitForest, data.test$yc)
m3_perf <- performance(m3_pred, "tpr", "fpr")

#plot variable importance
varImpPlot(m3, main="Random Forest: Variable Importance")

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 15:

# Model Performance plot
plot(m3_perf,colorize=TRUE, lwd=2, main = "m3 ROC: Random Forest", col = "blue")
lines(x=c(0, 1), y=c(0, 1), col="red", lwd=1, lty=3);
lines(x=c(1, 0), y=c(0, 1), col="green", lwd=1, lty=4)

m3_AUROC <- round(performance(m3_pred, measure = "auc")@y.values[[1]]*100, 2)
m3_auroc
cat("AUROC: ",m3_AUROC)

# Random Forest  : 97% accuracy 

#---------------------------------------------------------------------------------------------------------------
# Solution to Question 16:

# Clearly as observed, RF is performing better than LR in predicting our response variable yc

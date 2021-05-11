### ML Final Project Analysis ###
# Name: Olivia Delau & Jing Xu

### Packages you may need ######################################################
rm(list=ls())

library(tidyverse)
library(dplyr)
library(tableone)
library(glmnet)
library(tree)
library(randomForest)
library(rfUtilities)
library(ROCR)
library(MASS)
library(e1071)


#### Import data ##############################################################
setwd("/Users/jingxu/Desktop/J-O_ML_Project_2021") # set to your working directory
getwd()
stroke <- read.csv("stroke_ml.csv") #import cleaned dataset 
summary(stroke)
head(stroke)

#Factor variables
factors <- c("gender", "hypertension", "heart_disease", 
             "ever_married", "work_type", "residence_type",
             "smoking_status", "stroke")
stroke[factors] <- lapply(stroke[factors], factor) #make factors 

#### Make Table 1 #####################################################################
variables <- c("gender", "age", "hypertension", "heart_disease", "ever_married", "work_type", 
               "residence_type", "avg_glucose_level", "bmi", "smoking_status")
table1 <- CreateTableOne(vars = variables, 
                         strata = "stroke", 
                         data = stroke, 
                         factorVars = factors, 
                         addOverall = T,
                         test = F)
# table1 <- print(table1, exact = "stage", quote = FALSE, noSpaces = TRUE, printToggle = FALSE)
## Save to a CSV file
# write.csv(table1, file = "table1.csv")

#### Logistic regression ##############################################################

set.seed(0)
#make 5-folds
stroke_shuffle <- stroke[sample(nrow(stroke)),] #randomly shuffle data
folds <- cut(seq(1, nrow(stroke_shuffle)), breaks=5, labels=FALSE) #make 10 equally sized folds
# accur_lmod <- c() # store accuracy
auc_lmod <- c() #store auc
for(i in 1:5){
  #make testing and training data
  test_indexes <- which(folds==i,arr.ind=TRUE) #segment the data
  test <- stroke_shuffle[test_indexes, ] #test set
  train <- stroke_shuffle[-test_indexes, ] #train set
  lmod <- glm(stroke ~ gender +age + hypertension + heart_disease
              + ever_married + residence_type + avg_glucose_level + 
                bmi + smoking_status, train, family = "binomial") #apply logistic reg
  #pred_stroke <- rep(0, nrow(test))
   prob_stroke <- predict(lmod, newdata = test, type = "response")
   
   #AUC value
   pred <- prediction(prob_stroke, test$stroke)
   perf <- performance(pred, measure = "auc")
   auc_lmod[i] <- perf@y.values[[1]] 
   
  # pred_stroke[prob_stroke > 0.5] <- 1
  # accur_lmod[i] <- mean(test$stroke == pred_stroke) #accuracy 
  assign(paste("prob", i, sep = "_"), prob_stroke) #assign posterior prob to prob_i
  assign(paste("label", i, sep = "_"), test$stroke) #store true labels to label_i
}
# mean_accur_lmod <- mean(accur_lmod) #gives you the mean accuracy
# mean_accur_lmod

predictions_lmod <- data.frame(prob_1, prob_2, prob_3, prob_4, prob_5) #Predicted values
labels_lmod <- data.frame(label_1, label_2, label_3, label_4, label_5) #true labels 

mean_auc_lmod <- mean(auc_lmod) #mean auc

# Make ROC curve
pred_lmod <- prediction(predictions_lmod, labels_lmod)
perf_lmod <- performance(pred_lmod, "tpr", "fpr")
plot(perf_lmod,
     avg= "threshold",
     colorize=TRUE,
     lwd= 3,
     main= "ROC Curve for Logistic Regression Model") # ROC curve for logmod 
abline(a=0, b=1) # The completely random 

# Find AUC value
auc_perf_lmod <- performance(pred_lmod, measure = "auc")
mean_auc_lmod <- sum(auc_perf_lmod@y.values[[1]], auc_perf_lmod@y.values[[2]],
                      auc_perf_lmod@y.values[[3]], auc_perf_lmod@y.values[[4]],
                      auc_perf_lmod@y.values[[5]])/5 #gives you the mean AUC value

# Find the optimal cutoff + sensitivity and specificity
opt.cut = function(perf, pred){
  cut.ind = mapply(FUN=function(x, y, p){
    d = (x - 0)^2 + (y-1)^2
    ind = which(d == min(d))
    c(sensitivity = y[[ind]], specificity = 1-x[[ind]], 
      cutoff = p[[ind]])
  }, perf@x.values, perf@y.values, pred@cutoffs)
} # define opt cut function
opt_lmod <- opt.cut(perf_lmod, pred_lmod)
opt_sens_lmod <- mean(opt_lmod[1,])
opt_spec_lmod <- mean(opt_lmod[2,])
opt_cut_lmod <- mean(opt_lmod[3,])

# Make Accuracy Plot 
acc_perf_lmod = performance(pred_lmod, measure = "acc")
plot(acc_perf_lmod, 
     avg= "vertical",
     main  = "Average Accuracy")

#Find accuracy for opt cutoff  
avg_cut_lmod <- (slot(acc_perf_lmod, "x.values")[[1]] +
                  slot(acc_perf_lmod, "x.values")[[2]] +
                  slot(acc_perf_lmod, "x.values")[[3]] + 
                  slot(acc_perf_lmod, "x.values")[[4]] +
                  slot(acc_perf_lmod, "x.values")[[5]])/5
ind_cut_lmod = 176
acc_lmod = slot(acc_perf_lmod, "y.values")[[1]][ind_cut_lmod]

# Sensitivity vs. Specificity
sen_perf_lmod <- performance(pred_lmod, "sens", "spec")
plot(sen_perf_lmod,
     avg = "threshold",
     colorize = T,
     main = "Average Sensitivity vs. Average Specificity")

print(c(
  "AUC" = mean_auc_lmod,
  "Sensitivity" = opt_sens_lmod,
  "Specificity" = opt_spec_lmod,
  "opt cut-off" = opt_cut_lmod,
  "Accuracy" = acc_lmod
))

### LDA ######################################################################
set.seed(0)
#make 5-folds
stroke_shuffle <- stroke[sample(nrow(stroke)),] #randomly shuffle data
folds <- cut(seq(1, nrow(stroke_shuffle)), breaks=5, labels=FALSE) #make 10 equally sized folds
auc_lda <- c() #store auc
for(i in 1:5){
  #make testing and training data
  test_indexes <- which(folds==i,arr.ind=TRUE) #segment the data
  test <- stroke_shuffle[test_indexes, ] #test set
  train <- stroke_shuffle[-test_indexes, ] #train set
  lda_fit <- lda(stroke ~ gender +age + hypertension + heart_disease
                 + ever_married + residence_type + avg_glucose_level + 
                   bmi + smoking_status, data = train) #apply lda
  lda_pred=predict(lda_fit, test)
  
  #AUC value
  pred <- prediction(lda_pred$posterior[,2], test$stroke)
  perf <- performance(pred, measure = "auc")
  auc_lda[i] <- perf@y.values[[1]] 
  
  assign(paste("prob", i, sep = "_"), lda_pred$posterior[,2]) #assign posterior prob to prob_i
  assign(paste("label", i, sep = "_"), test$stroke) #store true labels to label_i
}

predictions_lda <- data.frame(prob_1, prob_2, prob_3, prob_4, prob_5)
labels_lda <- data.frame(label_1, label_2, label_3, label_4, label_5)

mean_auc_lda <- mean(auc_lda)

#ROC plot
pred_lda <- prediction(predictions_lda, labels_lda)
perf_lda <- performance(pred_lda, "tpr", "fpr")
plot(perf_lda,
     avg= "threshold",
     colorize=TRUE,
     lwd= 3,
     main= "ROC Curve for LDA Model") # ROC curve for lda
abline(a=0, b=1) # The completely random line

#AUC value
auc_perf_lda <- performance(pred_lda, measure = "auc")
mean_auc_lda <- sum(auc_perf_lda@y.values[[1]], auc_perf_lda@y.values[[2]],
                     auc_perf_lda@y.values[[3]], auc_perf_lda@y.values[[4]],
                     auc_perf_lda@y.values[[5]])/5

# Find the optimal cutoff + sensitivity and specificity
opt_lda <- opt.cut(perf_lda, pred_lda)
opt_sens_lda <- mean(opt_lda[1,])
opt_spec_lda <- mean(opt_lda[2,])
opt_cut_lda <- mean(opt_lda[3,])

# Make Accuracy Plot 
acc_perf_lda = performance(pred_lda, measure = "acc")
plot(acc_perf_lda, 
     avg= "vertical",
     main  = "Average Accuracy")

#Find accuracy for opt cutoff  
avg_cut_lda <- (slot(acc_perf_lda, "x.values")[[1]] +
                  slot(acc_perf_lda, "x.values")[[2]] +
                  slot(acc_perf_lda, "x.values")[[3]] + 
                  slot(acc_perf_lda, "x.values")[[4]] +
                  slot(acc_perf_lda, "x.values")[[5]])/5
ind_cut_lda = 188
acc_lda = slot(acc_perf_lda, "y.values")[[1]][ind_cut_lda]

# Sensitivity vs. Specificity
sen_perf_lda <- performance(pred_lda, "sens", "spec")
plot(sen_perf_lda,
     avg = "threshold",
     colorize = T,
     main = "Average Sensitivity vs. Average Specificity for LDA Model")

print(c(
  "AUC" = mean_auc_lda,
  "Sensitivity" = opt_sens_lda,
  "Specificity" = opt_spec_lda,
  "opt cut-off" = opt_cut_lda,
  "Accuracy" = acc_lda
))

### QDA ##################################################################
set.seed(0)
#make 5-folds
stroke_shuffle <- stroke[sample(nrow(stroke)),] #randomly shuffle data
folds <- cut(seq(1, nrow(stroke_shuffle)), breaks=5, labels=FALSE) #make 10 equally sized folds
auc_qda <- c() #store auc
for(i in 1:5){
  #make testing and training data
  test_indexes <- which(folds==i,arr.ind=TRUE) #segment the data
  test <- stroke_shuffle[test_indexes, ] #test set
  train <- stroke_shuffle[-test_indexes, ] #train set
  qda_fit <- qda(stroke ~ gender +age + hypertension + heart_disease
                 + ever_married + residence_type + avg_glucose_level + 
                   bmi + smoking_status, data = train) #apply qda
  qda_pred=predict(qda_fit, test)
  
  #AUC value
  pred <- prediction(qda_pred$posterior[,2], test$stroke)
  perf <- performance(pred, measure = "auc")
  auc_qda[i] <- perf@y.values[[1]] 
  
  assign(paste("prob", i, sep = "_"), qda_pred$posterior[,2]) #assign posterior prob to prob_i
  assign(paste("label", i, sep = "_"), test$stroke) #store true labels to label_i
}

predictions_qda <- data.frame(prob_1, prob_2, prob_3, prob_4, prob_5)
labels_qda <- data.frame(label_1, label_2, label_3, label_4, label_5)

mean_auc_qda <- mean(auc_qda)

#ROC plot
pred_qda <- prediction(predictions_qda, labels_qda)
perf_qda <- performance(pred_qda, "tpr", "fpr")
plot(perf_qda,
     avg= "threshold",
     colorize=TRUE,
     lwd= 3,
     main= "ROC Curve for QDA Model") # ROC curve for qda
abline(a=0, b=1) # The completely random line

# Find the optimal cutoff + sensitivity and specificity
opt_qda <- opt.cut(perf_qda, pred_qda)
opt_sens_qda <- mean(opt_qda[1,])
opt_spec_qda <- mean(opt_qda[2,])
opt_cut_qda <- mean(opt_qda[3,])


# Make Accuracy Plot 
acc_perf_qda = performance(pred_qda, measure = "acc")
plot(acc_perf_qda, 
     avg= "vertical",
     main  = "Average Accuracy")

#Find accuracy for opt cutoff  
avg_cut_qda <- (slot(acc_perf_qda, "x.values")[[1]] +
                   slot(acc_perf_qda, "x.values")[[2]] +
                   slot(acc_perf_qda, "x.values")[[3]] + 
                   slot(acc_perf_qda, "x.values")[[4]] +
                   slot(acc_perf_qda, "x.values")[[5]])/5
ind_cut_qda = 197
acc_qda = slot(acc_perf_qda, "y.values")[[1]][ind_cut_qda]

# Sensitivity vs. Specificity
sen_perf_qda <- performance(pred_qda, "sens", "spec")
plot(sen_perf_qda,
     avg = "threshold",
     colorize = T,
     main = "Average Sensitivity vs. Average Specificity for QDA Model")

print(c(
  "AUC" = mean_auc_qda,
  "Sensitivity" = opt_sens_qda,
  "Specificity" = opt_spec_qda,
  "opt cut-off" = opt_cut_qda,
  "Accuracy" = acc_qda
))

### SVM ################################################################################

# set.seed(0)
# tune.out=tune(svm, stroke ~ gender +age + hypertension + heart_disease
#               + ever_married + residence_type + avg_glucose_level + 
#                 bmi + smoking_status, 
#               data=train,
#               kernel="radial",
#               scale = T,
#               ranges=list(cost=c(0.1,1,10,100,1000),
#                             gamma=c(0.5,1,2,3,4)))
# summary(tune.out)
# svm.fit <- svm(stroke ~ gender +age + hypertension + heart_disease
#                + ever_married + residence_type + avg_glucose_level + 
#                  bmi + smoking_status, 
#                data=train, 
#                kernel="radial", 
#                gamma=3,
#                cost =10,
#                scale = T)
# true=test$stroke
# pred=predict(svm.fit, newdata=test)
# mean(true == pred)
# table(true, pred)


set.seed(0)
#make 5-folds
stroke_shuffle <- stroke[sample(nrow(stroke)),] #randomly shuffle data
folds <- cut(seq(1, nrow(stroke_shuffle)), breaks=5, labels=FALSE) #make 10 equally sized folds
accur_svm <- c() # store accuracy
auc_svm <- c() #store auc
sens_svm <- c() #store sensitivity
spec_svm <- c() #store specificity
for(i in 1:5){
  #make testing and training data
  test_indexes <- which(folds==i,arr.ind=TRUE) #segment the data
  test <- stroke_shuffle[test_indexes, ] #test set
  train <- stroke_shuffle[-test_indexes, ] #train set
  svmfit <- svm(stroke ~ gender +age + hypertension + heart_disease
             + ever_married + residence_type + avg_glucose_level + 
               bmi + smoking_status, data=train, kernel="linear", cost=3,
             scale=T, decision.values = T)
  prediction_table <- predict(svmfit, test)
  accur_svm[i] <- mean(test$stroke == prediction_table) #store accuracy 
  
  #Sensitivity and Specificity
  conf_matrix <- table(prediction_table, test$stroke)
  sens_svm[i] <- conf_matrix[2,2]/(conf_matrix[2,2] + conf_matrix[1,2]) #sensitivity
  spec_svm[i] <- conf_matrix[1,1]/(conf_matrix[1,1] + conf_matrix[2,1]) #specificity
  
  pred_auc <- attributes(predict(svmfit, test, decision.values = T))$decision.values
  
  pred <- prediction(pred_auc, test$stroke)
  perf <- performance(pred, measure = "auc")
  auc_svm[i] <- perf@y.values[[1]] 
  
  assign(paste("prob", i, sep = "_"), pred_auc) #assign posterior prob to prob_i
  assign(paste("label", i, sep = "_"), test$stroke) #store true labels to label_i
}

mean_accur_svm <- mean(accur_svm) #Mean accuracy
mean_sens_svm <- mean(sens_svm) #mean sensitivity
mean_spec_svm <- mean(spec_svm) #mean specificity

mean_auc_svm <- mean(auc_svm)

print(c(
  "Mean AUC" = mean_auc_svm,
  "Mean Accuracy" = mean_accur_svm,
  "Mean Sensitivity" = mean_sens_svm, 
  "Mean Specificity" = mean_spec_svm
))

predictions_svm <- data.frame(prob_1, prob_2, prob_3, prob_4, prob_5) #Predicted values
labels_svm <- data.frame(label_1, label_2, label_3, label_4, label_5) #true labels 

pred_svm <- prediction(predictions_svm, labels_svm)
perf_svm <- performance(pred_svm, "tpr", "fpr")
plot(perf_svm,
     avg= "threshold",
     colorize=TRUE,
     lwd= 3,
     main= "ROC Curve") # ROC curve for qda
abline(a=0, b=1) # The completely random line



### RF ########################################################################################
set.seed(0)
#make 5-folds
stroke_shuffle <- stroke[sample(nrow(stroke)),] #randomly shuffle data
folds <- cut(seq(1, nrow(stroke_shuffle)), breaks=5, labels=FALSE) #make 10 equally sized folds
auc_rf <- c() #store auc values
for(i in 1:5){
  #make testing and training data
  test_indexes <- which(folds==i,arr.ind=TRUE) #segment the data
  test <- stroke_shuffle[test_indexes, ] #test set
  train <- stroke_shuffle[-test_indexes, ] #train set
  rf.stroke = randomForest(stroke ~ gender +age + hypertension + heart_disease
                           + ever_married + residence_type + avg_glucose_level + 
                             bmi + smoking_status, data=train, ntree=100 ,mtry=3, importance =TRUE)
  prediction_table <- predict(rf.stroke, test)
  pred_ROC <- predict(rf.stroke, test, type="prob")
  
  #AUC value
  pred <- prediction(pred_ROC[,2], test$stroke)
  perf <- performance(pred, measure = "auc")
  auc_rf[i] <- perf@y.values[[1]] 
  
  assign(paste("prob", i, sep = "_"), pred_ROC[,2]) #assign prob to prob_i
  assign(paste("label", i, sep = "_"), test$stroke) #store true labels to label_i
}

mean_auc_rf <- mean(auc_rf) #Mean rf


predictions_rf <- data.frame(prob_1, prob_2, prob_3, prob_4, prob_5) #Predicted values
labels_rf <- data.frame(label_1, label_2, label_3, label_4, label_5) #true labels 

# Make ROC curve
pred_rf <- prediction(predictions_rf, labels_rf)
perf_rf <- performance(pred_rf, "tpr", "fpr")
plot(perf_rf,
     avg= "threshold",
     colorize=F,
     lwd= 1,
     main= "ROC Curve") # ROC curve for rf
abline(a=0, b=1) # The completely random 

# Find the optimal cutoff + sensitivity and specificity
opt_rf <- opt.cut(perf_rf, pred_rf)
opt_sens_rf <- mean(opt_rf[1,])
opt_spec_rf <- mean(opt_rf[2,])
opt_cut_rf <- mean(opt_rf[3,])

# Make Accuracy Plot 
acc_perf_rf = performance(pred_rf, measure = "acc")
plot(acc_perf_rf, 
     avg= "vertical",
     main  = "Average Accuracy")

#Find accuracy for opt cutoff  
avg_cut_rf <- (slot(acc_perf_rf, "x.values")[[1]] +
                   slot(acc_perf_rf, "x.values")[[2]] +
                   slot(acc_perf_rf, "x.values")[[3]] + 
                   slot(acc_perf_rf, "x.values")[[4]] +
                   slot(acc_perf_rf, "x.values")[[5]])/5
ind_cut_rf = 36
acc_rf = slot(acc_perf_rf, "y.values")[[1]][ind_cut_rf]

print(c(
  "AUC" = mean_auc_rf,
  "Sensitivity" = opt_sens_rf,
  "Specificity" = opt_spec_rf,
  "opt cut-off" = opt_cut_rf,
  "Accuracy" = acc_rf
))
### All ROC curve ##################################################################

plot(perf_lmod,
     avg= "threshold",
     lwd= 1,
     col = "red") #lmod curve
plot(perf_lda,
     avg= "threshold",
     add = T,
     lwd= 1,
     col = "blue") #lda curve
plot(perf_qda,
     avg= "threshold",
     add = T,
     lwd= 1,
     col= "purple") #qda curve
plot(perf_rf,
     avg= "threshold",
     add=T,
     lwd= 1,
     col= "green") #rf curve
plot(perf_svm,
     avg= "threshold",
     add = T,
     lwd= 1,
     col= "orange") #svm curve
abline(a=0, b=1, )
legend("bottomright", c("Log Regression", "LDA", "QDA", "Random Forest", "SVM"),
       lty = 1, col = c("red", "blue", "purple", "green", "orange"), bty = "n", 
       inset = c(0, 0.015))











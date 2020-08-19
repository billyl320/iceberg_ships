#r simple svm model

rm(list=ls())


library(xtable) #for table creation for latex
library(MASS)#for qda
library(plyr)#for obtaining means by factor
library(e1071)#for tune
library(caret)#for more info on training rf
library(randomForest)#for more info on training rf

#loading data
labs = read.table('LABS_ice_boat.txt', sep=',', header=TRUE)

#ei load
ei = read.table('AVE_ice_boat.txt', sep=',', header=TRUE)

#other shape metrics
shapes<-read.table('ice_boat_AVE_SHAPES.txt', sep=',', header=TRUE)

#1 = iceberg; 0 = boat
labs2<-as.factor(unlist( labs ))
labs2<-unname(labs2)
#
#combining data
temp<-as.data.frame(cbind(ei, ei[,1]/(ei[,1]+ei[,2]), shapes, shapes[,6]/(shapes[,6]+shapes[,7]) ))#,
#                          ei[,3]/(ei[,3]+ei[,4]), shapes))
#colnames(temp)[5]<-'sp_hh'
#colnames(temp)[6]<-'sp_hv'
colnames(temp)[3]<-'sp'
colnames(temp)[18]<-'sp_rect'


#check distributions of the observations per class
#remove some variables
not_use<-c(8,11:17)+1

#data prep
test<-as.data.frame(cbind(labs2, temp))
colnames(test)[1]<-"labs_svm"

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)


#summary of variables
summary_1<-sapply(test[keep1,-not_use],margins=2, summary)

summary_0<-sapply(test[keep2,-not_use],margins=2, summary)

sum_mat_1<-matrix(unlist(summary_1)[-c(1,2)],ncol=6, byrow=TRUE)
rownames(sum_mat_1)<-names(summary_1)[-1]
colnames(sum_mat_1)<-names(summary_1$sp)

sum_mat_0<-matrix(unlist(summary_0)[-c(1,2)], ncol=6, byrow=TRUE)
rownames(sum_mat_0)<-names(summary_0)[-1]
colnames(sum_mat_0)<-names(summary_0$sp)


#report summaries 
xtable(sum_mat_1)

xtable(sum_mat_0)


#now let's tune the rf model using 10-folds on t-set and validaiton

#set seed
set.seed(80636)

#doing testing validation split prep
keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

#ensuring proportions remain the same between testing and validation sets
valid_1<-sample(keep1, floor(length(keep1)*0.20) )
valid_2<-sample(keep2, floor(length(keep2)*0.20))

valid<-c(valid_1, valid_2)

train<-test[-valid,]

#cv setup
tc <- trainControl(method='cv',
                  number = 10,
                  search='grid')

grid <- expand.grid(mtry=c(1:6))

#perform cv random forest
tune.out<-train(as.factor(labs_svm) ~.,
          data=train[,-not_use],
          method='rf',
          trControl = tc,
          tuneGrid=grid)

#print results 
print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

varImp(tune.out)

#output accuracy on training data
ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

#confusion matrix
confusionMatrix(ypred, test$labs_svm[-valid])

#setup matrix to collect scores
measures_test<-matrix(nrow=2, ncol=3, data=0 )
rownames(measures_test)<-c('0', '1')
colnames(measures_test)<-c("Precision", "Recall", "F-1 Score")

#collecting measures
precision <- posPredValue(ypred, test$labs_svm[-valid], positive="0")
recall <- sensitivity(ypred, test$labs_svm[-valid], positive="0")
F1 <- (2 * precision * recall) / (precision + recall)
measures_test[1,1]<-precision
measures_test[1,2]<-recall
measures_test[1,3]<-F1

precision <- posPredValue(ypred, test$labs_svm[-valid], positive="1")
recall <- sensitivity(ypred, test$labs_svm[-valid], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
measures_test[2,1]<-precision
measures_test[2,2]<-recall
measures_test[2,3]<-F1

xtable(measures_test)

colMeans(measures_test)

#collecting accuracy on validation data
ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

#confusion matrix for validation data
confusionMatrix(ypred, test$labs_svm[valid])

#matrix for validation data measures
measures_valid<-matrix(nrow=2, ncol=3, data=0 )
rownames(measures_valid)<-c('0', '1')
colnames(measures_valid)<-c("Precision", "Recall", "F-1 Score")

#collecting measures
precision <- posPredValue(ypred, test$labs_svm[valid], positive="0")
recall <- sensitivity(ypred, test$labs_svm[valid], positive="0")
F1 <- (2 * precision * recall) / (precision + recall)
measures_valid[1,1]<-precision
measures_valid[1,2]<-recall
measures_valid[1,3]<-F1

precision <- posPredValue(ypred, test$labs_svm[valid], positive="1")
recall <- sensitivity(ypred, test$labs_svm[valid], positive="1")
F1 <- (2 * precision * recall) / (precision + recall)
measures_valid[2,1]<-precision
measures_valid[2,2]<-recall
measures_valid[2,3]<-F1

xtable(measures_valid)

colMeans(measures_valid)
#

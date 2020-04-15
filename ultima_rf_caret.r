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

#ei

#ei_HH = read.table('HH_ice_boat.txt', sep=',', header=TRUE)
#ei_HV = read.table('HV_ice_boat.txt', sep=',', header=TRUE)
#ei = cbind(ei_HH, ei_HV)
ei = read.table('AVE_ice_boat.txt', sep=',', header=TRUE)
#colnames(ei)<-c(paste(colnames(ei)[1:2],'_HH', sep=''),
#                paste(colnames(ei)[1:2],'_HV', sep=''))

#other shape metrics
#shapes_HH<-read.table('ice_boat_HH_SHAPES.txt', sep=',', header=TRUE)
#shapes_HV<-read.table('ice_boat_HV_SHAPES.txt', sep=',', header=TRUE)
#shapes<-cbind(shapes_HH, shapes_HV)
shapes<-read.table('ice_boat_AVE_SHAPES.txt', sep=',', header=TRUE)

#colnames(shapes)<-c(paste(colnames(shapes_HH),'_HH', sep=''),
#                    paste(colnames(shapes_HH),'_HV', sep=''))


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
not_use<-c(8,11:17)+1

test<-as.data.frame(cbind(labs2, temp))
colnames(test)[1]<-"labs_svm"

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

summary_1<-sapply(test[keep1,-not_use],margins=2, summary)

summary_0<-sapply(test[keep2,-not_use],margins=2, summary)

sum_mat_1<-matrix(unlist(summary_1)[-c(1,2)],ncol=6, byrow=TRUE)
rownames(sum_mat_1)<-names(summary_1)[-1]
colnames(sum_mat_1)<-names(summary_1$sp)

sum_mat_0<-matrix(unlist(summary_0)[-c(1,2)], ncol=6, byrow=TRUE)
rownames(sum_mat_0)<-names(summary_0)[-1]
colnames(sum_mat_0)<-names(summary_0$sp)

xtable(sum_mat_1)

xtable(sum_mat_0)


#now let's tune the rf model using 10-folds on t-set and validaiton

set.seed(80636)

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

valid_1<-sample(keep1, floor(length(keep1)*0.20) )
valid_2<-sample(keep2, floor(length(keep2)*0.20))

valid<-c(valid_1, valid_2)

train<-test[-valid,]

tc <- trainControl(method='cv',
                  number = 10,
                  search='grid')

grid <- expand.grid(mtry=c(1:6))

tune.out<-train(as.factor(labs_svm) ~.,
          data=train[,-not_use],
          method='rf',
          trControl = tc,
          tuneGrid=grid)

print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

varImp(tune.out)

ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

confusionMatrix(ypred, test$labs_svm[-valid])

measures_test<-matrix(nrow=2, ncol=3, data=0 )
rownames(measures_test)<-c('0', '1')
colnames(measures_test)<-c("Precision", "Recall", "F-1 Score")

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

ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

confusionMatrix(ypred, test$labs_svm[valid])


measures_valid<-matrix(nrow=2, ncol=3, data=0 )
rownames(measures_valid)<-c('0', '1')
colnames(measures_valid)<-c("Precision", "Recall", "F-1 Score")

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

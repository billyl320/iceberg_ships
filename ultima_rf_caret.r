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
temp<-as.data.frame(cbind(ei, ei[,1]/(ei[,1]+ei[,2]), shapes ))#,
#                          ei[,3]/(ei[,3]+ei[,4]), shapes))
#colnames(temp)[5]<-'sp_hh'
#colnames(temp)[6]<-'sp_hv'
colnames(temp)[3]<-'sp'


#check distributions of the observations per class
not_use<-c(8,11:17)+1

test<-as.data.frame(cbind(labs2, temp))
colnames(test)[1]<-"labs_svm"

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

summary_1<-sapply(test[keep1,-not_use],margins=2, summary)

summary_0<-sapply(test[keep2,-not_use],margins=2, summary)

sum_mat_1<-matrix(unlist(summary_1)[-c(1,2)], nrow=9, ncol=6, byrow=TRUE)
rownames(sum_mat_1)<-names(summary_1)[-1]
colnames(sum_mat_1)<-names(summary_1$sp)

sum_mat_0<-matrix(unlist(summary_0)[-c(1,2)], nrow=9, ncol=6, byrow=TRUE)
rownames(sum_mat_0)<-names(summary_0)[-1]
colnames(sum_mat_0)<-names(summary_0$sp)

xtable(sum_mat_1)

xtable(sum_mat_0)

#simple model to check if it is possible to do simple model
rf.fit<-randomForest(as.factor(test$labs_svm) ~.,
              data=test[,-not_use])#,
              #sampsize=make.size(as.factor(test$labs_svm)))

#predicting
rf.pred=predict(rf.fit, test)
rf.class = rf.pred

#test
table(rf.class, test$labs_svm)
#overall classification rate for training
mean(rf.class==test$labs_svm)

#now let's tune the svm model using 10-folds on t-set and validaiton

set.seed(80636)

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

valid_1<-sample(keep1, floor(length(keep1)*0.30) )
valid_2<-sample(keep2, floor(length(keep2)*0.30))

valid<-c(valid_1, valid_2)

tc <- trainControl(method='cv',
                  number = 10,
                  search='random')

tune.out<-train(as.factor(labs_svm) ~.,
          data=test[,-not_use],
          method='rf',
          trControl = tc)

print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

confusionMatrix(ypred, test$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

confusionMatrix(ypred, test$labs_svm[valid])

#####################
## Sanity Checks
#####################

#sanity check - swap the 0s and 1s to ensure that predictions flip
index_0<-which(test$labs_svm==0)
index_1<-which(test$labs_svm==1)

sanity_labs<-test$labs_svm
sanity_labs[index_0]=1
sanity_labs[index_1]=0

head(sanity_labs)

ypred_sanity=predict(tune.out$finalModel ,test)
table(predict=ypred_sanity, truth=sanity_labs)
mean(ypred_sanity==sanity_labs)

confusionMatrix(ypred_sanity, sanity_labs)

# since the results swapped, we know that the
# algorithm isn't making some silly error.

#now let's 'confuse' the algorithm by changing some of the values
# we'll simuate values by swapping around random values
# between pairs of observations (number_obs_sanity each )

num_0<-length(index_0)
num_1<-length(index_1)

number_obs_sanity<-300

#getting pairs
vals_0<-sample(index_0,
               size=number_obs_sanity,
               replace=FALSE)
vals_1<-sample(index_1,
               size=number_obs_sanity,
               replace=FALSE)

#creating changed data set
changed_data<-test[,-not_use]



for(i in 1:number_obs_sanity){

    #variable to swap
    temp_var<-sample(2:10, 4)

    #placeholder for 0
    temp_val<-changed_data[vals_0[i], temp_var]

    #swapping 1 for 0
    changed_data[vals_0[i], temp_var]<- changed_data[vals_1[i], temp_var]

    #swapping 0 for 1
    changed_data[vals_1[i], temp_var]<- temp_var

}

#now predicting on the data using swapped observations and trained model

ypred=predict(tune.out$finalModel ,changed_data[-valid,])
table(predict=ypred, truth=changed_data$labs_svm[-valid])
mean(ypred==changed_data$labs_svm[-valid])

confusionMatrix(ypred, changed_data$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,changed_data[valid,])
table(predict=ypred, truth=changed_data$labs_svm[valid])
mean(ypred==changed_data$labs_svm[valid])

confusionMatrix(ypred, changed_data$labs_svm[valid])

#
#now let's do it for the most important triad of variables
# (e1, e2, and eccentricity)

#creating changed data set
changed_data<-test[,-not_use]


for(i in 1:number_obs_sanity){

    #variable to swap
    temp_var<-c(6:8)

    #placeholder for 0
    temp_val<-changed_data[vals_0[i], temp_var]

    #swapping 1 for 0
    changed_data[vals_0[i], temp_var]<- changed_data[vals_1[i], temp_var]

    #swapping 0 for 1
    changed_data[vals_1[i], temp_var]<- temp_var

}

#now predicting on the data using swapped observations and trained model

ypred=predict(tune.out$finalModel ,changed_data[-valid,])
table(predict=ypred, truth=changed_data$labs_svm[-valid])
mean(ypred==changed_data$labs_svm[-valid])

confusionMatrix(ypred, changed_data$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,changed_data[valid,])
table(predict=ypred, truth=changed_data$labs_svm[valid])
mean(ypred==changed_data$labs_svm[valid])

confusionMatrix(ypred, changed_data$labs_svm[valid])

#now let's do it for SPEI vars
# (white, black, sp)

#creating changed data set
changed_data<-test[,-not_use]


for(i in 1:number_obs_sanity){

    #variable to swap
    temp_var<-c(2:4)

    #placeholder for 0
    temp_val<-changed_data[vals_0[i], temp_var]

    #swapping 1 for 0
    changed_data[vals_0[i], temp_var]<- changed_data[vals_1[i], temp_var]

    #swapping 0 for 1
    changed_data[vals_1[i], temp_var]<- temp_var

}

#now predicting on the data using swapped observations and trained model

ypred=predict(tune.out$finalModel ,changed_data[-valid,])
table(predict=ypred, truth=changed_data$labs_svm[-valid])
mean(ypred==changed_data$labs_svm[-valid])

confusionMatrix(ypred, changed_data$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,changed_data[valid,])
table(predict=ypred, truth=changed_data$labs_svm[valid])
mean(ypred==changed_data$labs_svm[valid])

confusionMatrix(ypred, changed_data$labs_svm[valid])

#now let's do it for remaining variables
# (circ, black_box, white_box)

#creating changed data set
changed_data<-test[,-not_use]


for(i in 1:number_obs_sanity){

    #variable to swap
    temp_var<-c(5,9,10)

    #placeholder for 0
    temp_val<-changed_data[vals_0[i], temp_var]

    #swapping 1 for 0
    changed_data[vals_0[i], temp_var]<- changed_data[vals_1[i], temp_var]

    #swapping 0 for 1
    changed_data[vals_1[i], temp_var]<- temp_var

}

#now predicting on the data using swapped observations and trained model

ypred=predict(tune.out$finalModel ,changed_data[-valid,])
table(predict=ypred, truth=changed_data$labs_svm[-valid])
mean(ypred==changed_data$labs_svm[-valid])

confusionMatrix(ypred, changed_data$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,changed_data[valid,])
table(predict=ypred, truth=changed_data$labs_svm[valid])
mean(ypred==changed_data$labs_svm[valid])

confusionMatrix(ypred, changed_data$labs_svm[valid])


#now let's randomly select any number of variables

#creating changed data set
changed_data<-test[,-not_use]


for(i in 1:number_obs_sanity){

    num_swap<-c(1:9, 1)

    #variable to swap
    temp_var<-sample(2:10, num_swap)

    #placeholder for 0
    temp_val<-changed_data[vals_0[i], temp_var]

    #swapping 1 for 0
    changed_data[vals_0[i], temp_var]<- changed_data[vals_1[i], temp_var]

    #swapping 0 for 1
    changed_data[vals_1[i], temp_var]<- temp_var

}

#now predicting on the data using swapped observations and trained model

ypred=predict(tune.out$finalModel ,changed_data[-valid,])
table(predict=ypred, truth=changed_data$labs_svm[-valid])
mean(ypred==changed_data$labs_svm[-valid])

confusionMatrix(ypred, changed_data$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,changed_data[valid,])
table(predict=ypred, truth=changed_data$labs_svm[valid])
mean(ypred==changed_data$labs_svm[valid])

confusionMatrix(ypred, changed_data$labs_svm[valid])


###############################################
#repeating to check if results are the same
#using mtry=1

set.seed(9400)

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

valid_1<-sample(keep1, floor(length(keep1)*0.30) )
valid_2<-sample(keep2, floor(length(keep2)*0.30))

valid<-c(valid_1, valid_2)

tc <- trainControl(method='cv',
                  number = 10,
                  search='grid')

grid <- expand.grid(mtry=1)

tune.out<-train(as.factor(labs_svm) ~.,
          data=test[,-not_use],
          method='rf',
          trControl = tc,
          tuneGrid=grid)

print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

confusionMatrix(ypred, test$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

confusionMatrix(ypred, test$labs_svm[valid])

set.seed(60662)

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

valid_1<-sample(keep1, floor(length(keep1)*0.30) )
valid_2<-sample(keep2, floor(length(keep2)*0.30))

valid<-c(valid_1, valid_2)

tc <- trainControl(method='cv',
                  number = 10,
                  search='grid')

grid <- expand.grid(mtry=1)

tune.out<-train(as.factor(labs_svm) ~.,
          data=test[,-not_use],
          method='rf',
          trControl = tc,
          tuneGrid=grid)

print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

confusionMatrix(ypred, test$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

confusionMatrix(ypred, test$labs_svm[valid])

set.seed(53978)

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

valid_1<-sample(keep1, floor(length(keep1)*0.30) )
valid_2<-sample(keep2, floor(length(keep2)*0.30))

valid<-c(valid_1, valid_2)

tc <- trainControl(method='cv',
                  number = 10,
                  search='grid')

grid <- expand.grid(mtry=1)

tune.out<-train(as.factor(labs_svm) ~.,
          data=test[,-not_use],
          method='rf',
          trControl = tc,
          tuneGrid=grid)

print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

confusionMatrix(ypred, test$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

confusionMatrix(ypred, test$labs_svm[valid])

set.seed(7404)

keep1<-which(test$labs_svm==1)
keep2<-which(test$labs_svm==0)

valid_1<-sample(keep1, floor(length(keep1)*0.30) )
valid_2<-sample(keep2, floor(length(keep2)*0.30))

valid<-c(valid_1, valid_2)

tc <- trainControl(method='cv',
                  number = 10,
                  search='grid')

grid <- expand.grid(mtry=1)

tune.out<-train(as.factor(labs_svm) ~.,
          data=test[,-not_use],
          method='rf',
          trControl = tc,
          tuneGrid=grid)

print(tune.out)

(tune.out$finalModel$importance)

varImp(tune.out$finalModel)

ypred=predict(tune.out$finalModel ,test[-valid,])
table(predict=ypred, truth=test$labs_svm[-valid])
mean(ypred==test$labs_svm[-valid])

confusionMatrix(ypred, test$labs_svm[-valid])

ypred=predict(tune.out$finalModel ,test[valid,])
table(predict=ypred, truth=test$labs_svm[valid])
mean(ypred==test$labs_svm[valid])

confusionMatrix(ypred, test$labs_svm[valid])

#repeating random partitions 100 times to see if results line up


set.seed(70172296)

acc_t<-c()
acc_v<-c()

for(i in 1:100){

  keep1<-which(test$labs_svm==1)
  keep2<-which(test$labs_svm==0)

  valid_1<-sample(keep1, floor(length(keep1)*0.30) )
  valid_2<-sample(keep2, floor(length(keep2)*0.30))

  valid<-c(valid_1, valid_2)

  tc <- trainControl(method='cv',
                    number = 10,
                    search='grid')

  grid <- expand.grid(mtry=1)

  tune.out<-train(as.factor(labs_svm) ~.,
            data=test[,-not_use],
            method='rf',
            trControl = tc,
            tuneGrid=grid)


  ypred=predict(tune.out$finalModel ,test[-valid,])
  acc_t[i]<-mean(ypred==test$labs_svm[-valid])


  ypred=predict(tune.out$finalModel ,test[valid,])
  acc_v[i]<-mean(ypred==test$labs_svm[valid])

}

summary(acc_t)

summary(acc_v)


#

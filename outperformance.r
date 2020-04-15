#outperformance rate calculation for RF over CNN
#CNNs from Li et.al, 2019

#testing
cnn_acc<-c(91.5, 93.72, 93.5, 94.9)

rf_acc<-c(rep(99.60, 4))

mean(rf_acc/cnn_acc)-1

#validation
cnn_acc<-c(86.51, 87.72, 84.83, 87.02)

rf_acc<-c(rep(95.71, 4))

mean(rf_acc/cnn_acc)-1


#

# Prepare data

set.seed(3)
x=matrix(rnorm(20*2), ncol=2)
y=c(rep(-1,10), rep(1,10))
x[y==1,]=x[y==1,] + 1
dat=data.frame(x=x, y=as.factor(y))

plot(x, col=(3-y))

# fit SVM
library(e1071)
svmfit=svm(y~., data=dat, kernel="linear", cost=10,scale=FALSE)

plot(svmfit, dat); svmfit$index

svmfit=svm(y~., data=dat, kernel="linear", cost=0.1,scale=FALSE)
plot(svmfit, dat); svmfit$index

# cross validation
set.seed(1)
tune.out=tune(svm,y~.,data=dat,
              kernel="linear",
              ranges=list(cost=c(0.001, 0.01, 0.1, 1,5,10,100)))

summary(tune.out)

bestmod=tune.out$best.model
summary(bestmod)

# prediction with SVM

xtest=matrix(rnorm(20*2), ncol=2)
ytest=sample(c(-1,1), 20, rep=TRUE)
xtest[ytest==1,]=xtest[ytest==1,] + 1
testdat=data.frame(x=xtest, y=as.factor(ytest))
ypred=predict(bestmod,testdat)
table(predict=ypred, truth=testdat$y)

svmfit=svm(y~., data=dat, kernel="linear", cost=.01,scale=FALSE)
ypred=predict(svmfit,testdat)
table(predict=ypred, truth=testdat$y)

# Complete split

x[y==1,]=x[y==1,]+0.5
plot(x, col=(y+5)/2, pch=19)

dat=data.frame(x=x, y=as.factor(y))
svmfit=svm(y~., data=dat, kernel="linear", cost=1e5)
summary(svmfit)

plot(svmfit, dat)

svmfit=svm(y~., data=dat, kernel="linear", cost=1)
summary(svmfit)

plot(svmfit,dat)

set.seed(1)
x=matrix(rnorm(200*2), ncol=2)
x[1:100,]=x[1:100,]+2
x[101:150,]=x[101:150,]-2
y=c(rep(1,150),rep(2,50))
dat=data.frame(x=x,y=as.factor(y)); plot(x, col=y)

# Using differen Kernels

train=sample(200,100)
svmfit=svm(y~., data=dat[train,], kernel="radial", gamma=1, cost=1)
plot(svmfit, dat[train,])

summary(svmfit)

svmfit=svm(y~., data=dat[train,], kernel="radial",gamma=1,cost=1e5)
plot(svmfit,dat[train,])

set.seed(1)
tune.out=tune(svm, y~., data=dat[train,], kernel="radial",
              ranges=list(cost=c(0.1,1,10,100,1000)))
# Need Working here
table(true=dat[-train,"y"],
      pred=predict(tune.out$best.model,newx=dat[-train,]))

# Using Sonar Data

library(dplyr)
library(mlbench)
# load Sonar dataset
data(Sonar)

# split train and test set
set.seed(123)
train <- sample.int(nrow(Sonar), size = floor(nrow(Sonar)*0.8))
train_X <- Sonar[train,]
test_X <- Sonar[-train,]

# Linear kernel
svmfit = svm(Class~., data = train_X, kernel = "linear")
summary(svmfit)

# Check train and test error
pred_train <- predict(svmfit, train_X)
mean(pred_train == train_X$Class)

pred_test <- predict(svmfit, test_X)
mean(pred_test == test_X$Class)

# Radial kernel
svmfit = svm(Class~., data = train_X, kernel = "radial")

# Check train and test error
pred_train <- predict(svmfit, train_X)
mean(pred_train == train_X$Class)

pred_test <- predict(svmfit, test_X)
mean(pred_test == test_X$Class)

# parameter tuning

tune.out=tune(svm, Class~., data=train_X, kernel="radial",
              ranges=list(cost=c(0.1,1,10,100,1000)))
summary(tune.out)
table(true=test_X$Class, pred=predict(tune.out$best.model,newdata=test_X))

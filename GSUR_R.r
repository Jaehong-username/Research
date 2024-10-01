setwd("D:\\OneDrive - Washington State University (email.wsu.edu)\\Dai LPRC Projects\\_Maureen Projects etc\\GSUR\\2024\\Jaehong_Lee\\Project\\Analysis")
#loading the data
library(caret)
library(pROC)
data = read.csv('n-back_data_GSUR-Project_n-back.csv', header = T)
View(data)

# Standardize predictor variables  normal distribution for every single variable
preproc <- preProcess(data[, -ncol(data)], method = c("scale"))

# Apply the transformation to the data
data_std <- predict(preproc, newdata = data)


#My method: cross valiadation train/test split data
#Now let's split the data on 7:3 ratio!
set.seed(200) #random!!
ss = sample(1:nrow(data), nrow(data)*0.7)
trainingData = data[ss,]
testingData = data[-ss,]
#View(trainingData)
#View(testingData)
#Now Let's test which ML algorithm works the best.
#THe feature of this data. Let's choose the target variable as binary

#first algo to test: Random Forest
library(randomForest)
library(dplyr)

colnames(trainingData)
print(trainingData)

#train the Random Forest Model
trainingDataSubset <- trainingData %>%
 dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
#Error: make nsure to use dplyr::select   and use comma, !!!



#Train the random Forest Model
randomForestModel1 <- randomForest(as.factor(trainingData$Healthy)~., data = trainingDataSubset, mtry = 3, ntree = 100)
randomForestModel1[["terms"]]
print(randomForestModel1) #default mtry 3 and ntree 100 has trees to groe lower error rate



#Now let's test the the model with the training data
trainPredictions1 <- predict(randomForestModel1, data = trainingDataSubset, type = "class")
trainPredictions1
#ConfusionMatrix
cftr1 <- table(Predicted = trainPredictions1, Actual = trainingData$Healthy)
confusionMatrix(cftr1)
#AUC
rocCurve11 <- roc(trainingData$Healthy, as.numeric(trainPredictions1))
aucValue11 <- auc(rocCurve11)
print(paste("AUC:", aucValue11))
plot(rocCurve11, col = "blue", main = "ROC Curve for Random Forest (Training Data)")




#now let's test the model!
testingDataSubset <- testingData %>%
  dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)

#Make predictions on the test set
testPredictions1 <- predict(randomForestModel1, newdata = testingDataSubset, type = "class")
#this will give predicted possibilities that each instance might belong to certain class
#testingResult1 <- data.frame(Prediction = ifelse(testPredictions1[, "1"] > 0.5, 1, 0))
#testingResult1 <- data.frame(ifelse(0.5 < rep(predictions1[,1], nrow(predictions1)), 0, 1))
cfte1 <- table(Predicted = testPredictions1, Actual = testingData$Healthy)
confusionMatrix(cfte1)

testPredictions1  # this already a vector

#calculate AUC for random Forest   roc function only inputs numeric!
rocCurve1 <- roc(testingData$Healthy, as.numeric(testPredictions1)) 
aucValue1 <- auc(rocCurve1)
print(paste("AUC:", aucValue1))
plot(rocCurve1, col = "blue", main = "ROC Curve for Random Forest")
#AUC 0.59 for Random Forest









#Validation data == testing data
#Next Let's do Logistic Regression with the binary target variable
library(ISLR2)

#Training the model
logisticRegressionModel1 <- glm(as.factor(trainingData$Healthy) ~., data = trainingDataSubset, family = 'binomial')
#logisticRegressionModel1 <- stepAIC(logisticRegressionModel1)
print(logisticRegressionModel1)
coef(logisticRegressionModel1)
summary(logisticRegressionModel1)$coef  #p value test: response Rate was a statistically significant!
#and some of these variables are not significant



#Now let's test the the model with the training data
trainPredictions2 <- predict(logisticRegressionModel1, data = trainingDataSubset, type = "response")
trainPredictions2
trainPredictions2Classified <- ifelse(trainPredictions2 > 0.5, 1, 0)
trainPredictions2Classified
#ConfusionMatrix
cftr2 <- table(Predicted = trainPredictions2Classified, Actual = trainingData$Healthy)
cftr2
confusionMatrix(cftr2)
#AUC
rocCurve22 <- roc(trainingData$Healthy, as.numeric(trainPredictions2))
aucValue22 <- auc(rocCurve22)
print(paste("AUC:", aucValue22))
plot(rocCurve22, col = "blue", main = "ROC Curve for Logistic Regression (Training Data)")
#0.796 AUC
#Accuracy 0.7182
#After stepAIC function Accuracy 0.7091    AUC: 0.75928    Without AIC function, yields a better result



#make predictions on the test set
testPredictions2 <- predict(logisticRegressionModel1, newdata = testingDataSubset, type = "response")
testPredictions2Classified <- ifelse(testPredictions2 > 0.5, 1, 0) # this code necessary!!!? to turn into classification
testPredictions2Classified
testingData$Healthy
#testingResult2 <- data.frame(Prediction = ifelse(predictions2 > 0.5, 1, 0))
cfte2 <- table(Predicted = testPredictions2Classified, Actual = testingData$Healthy)
confusionMatrix(cfte2)
#calculate AUC for logistic Regresion
#rocCurve2 <- roc(trainingData$Healthy, predictions2[, 1])
rocCurve2 <- roc(testingData$Healthy, as.numeric(testPredictions2Classified))
aucValue2 <- auc(rocCurve2)
print(paste("AUC:", aucValue2))
plot(rocCurve2, col = "blue", main = "ROC Curve for Logistic Regression")
#Without stepAIC function, Accuracy: 0.625,      AUC: 0.6357
#With step AIC function,   Accuracy 0.5417       AUC: 0.542857




#Actually in this case stepAIC() worsens the accuracy of the model
#improve the model with stepAIC() will search through the possible models and return the one with the lowest AIC
library("MASS")  #bc it achieves balance between goodness of fit and simplicity    trade off between model fit and complexity
improvedLogisticRegressionModel1 <- stepAIC(logisticRegressionModel1)
summary(improvedLogisticRegressionModel1)
#AIC: AIC=2k−2ln(L)
#k is the number of parameters in the model.
#L is the likelihood of the model, which represents how well the model fits the data.
#The Akaike Information Criterion (AIC) is a metric used in statistical model evaluation to balance the trade-off between model fit and complexity. 
#It helps to identify the best model among a set of candidate models by considering both the goodness of fit and the simplicity of the model.






#Next Let's use Naive Bayes
library(e1071)


naiveBayesModel1 <- naiveBayes(as.factor(trainingData$Healthy) ~., data = trainingDataSubset)
print(naiveBayesModel1)

#Let's test the model using the training data.    #have to use newdata param in this case
trainPredictions3 <- predict(naiveBayesModel1, newdata = trainingDataSubset, type = "class")
cftr3 <- table(Predicted = trainPredictions3, Actual = trainingData$Healthy)
confusionMatrix(cftr3)
#AUC
rocCurve33 <- roc(trainingData$Healthy, as.numeric(trainPredictions3))
aucValue33 <- auc(rocCurve33)
print(paste("AUC:", aucValue33))
plot(rocCurve33, col = "blue", main = "ROC Curve for Naive Bayes (Training Data")
#Naive Bayes:  Training Data    Accuracy: 0.7091       AUC:0.7122



#Let's test the model using the testing data
testPredictions3 <- predict(naiveBayesModel1, newdata = testingDataSubset, type = "class")
cfte3 <- table(Predicted = testPredictions3, Actual = testingData$Healthy)       
confusionMatrix(cfte3)
#colnames(predictions3)
#calculate AUC for Naive Bayes
rocCurve3 <- roc(testingData$Healthy, as.numeric(testPredictions3))
aucValue3 <- auc(rocCurve3)
print(paste("AUC:", aucValue3))
plot(rocCurve3, col = "blue", main = "ROC Curve for Naive Bayes")
#Naive Bayes:  Testing Data   Accuracy: 0.5833      AUC: 0.60714







#Next Let's do classification tree
#library(tree)
#library(ISLR2)
#treeModel1 <- tree(Healthy ∼., trainingDataSubset) #some unknown error!
library(rpart)
library(rpart.plot)

#Method class: for classification treeif th -e target var is binary. two different outcomes
#Method Anova: for regression tree if the var is continuous
treeModel1 <- rpart(as.factor(trainingData$Healthy) ~., data = trainingDataSubset, method = "class", cp = 0.001) 
# cp the lower set it will allow to grow more complex tree
prp(treeModel1, space = 4, split.cex =  1.5, nn.border.col = 0) #Another way of plotting the tree!!
rpart.plot(treeModel1, extra = 104) #Plotting the tree


#testing the tree model
predictions4 <- predict(treeModel1, data = trainingDataSubset, type = "class")
predictions4

#Working on the training data
cf4 <- table(Predicted = predictions4, Actual = as.factor(trainingData$Healthy))
cf4 #display
#error Error in xtfrm.data.frame(x) : cannot xtfrm data frames. Reason: because for predict I used data.frame 
#data frame changes the vector format.
#confusion mnartrix   library(caret)  library(e1071)
confusionMatrix(cf4)

#calculate AUC for Decision tree model
rocCurve4 <- roc(trainingData$Healthy, as.numeric(predictions4))
aucValue4 <- auc(rocCurve4)
print(paste("AUC:", aucValue4))
plot(rocCurve4, col = "blue", main = "ROC Curve for Classification Tree")
#Decision Tree: Training Data:   Accuracy: 0.7909   AUC: 0.79078



#Testing the data. Remember to use new data arfument!!
testPredictions4 <- predict(treeModel1, newdata = testingDataSubset, type = "class")
testPredictions4
cfte4 <- table(Predicted = testPredictions4, Actual = as.factor(testingData$Healthy))
cfte4
confusionMatrix(cfte4)
#calculate AUC for Decision tree model
rocCurve4 <- roc(testingData$Healthy, as.numeric(testPredictions4))
aucValue4 <- auc(rocCurve4)
print(paste("AUC:", aucValue4))
plot(rocCurve4, col = "blue", main = "ROC Curve for Classification Tree")
#Decision Tree: Testing Data:     Accuracy: 0.4375    AUC: 0.417857





#Now Let's do K nearest Neighbors!!
library(class)#knn function forms a prediction using a single command

#first input we need a matrix containing predictors
set.seed(1) # if several observations are tied as nearest neighbors, then R will randomly break the tie
knnModel1 <- knn(train = trainingDataSubset, test = testingDataSubset,
                 cl = trainingData$Healthy,k = round(sqrt(nrow(trainingDataSubset))))
knnModel1
plotPredictions5 <- trainingDataSubset %>%
  dplyr::select(-Healthy)
plotPredictions5Data <- data.frame(testingDataSubset, predicted = knnModel1)

library(ggplot2)
library(gridExtra)

#Question?? How to plot high demensional data onto the plot??? for visualization?
#testPredictions5 <- predict(knnModel1, newdata = testingDataSubset, type = "class")  why error???
#table(Predicted = testPredictions5, Actual = testingData$Healthy)

#Limitation Of the packages.
#Make sure to look for others packages
# cftr5 <- table(Predicted = knnModel1, Actual = trainingData$Healthy)################???????????????????????????????????????????????????
# confusionMatrix(cftr5)
# mean(knnModel1 == trainingData$Healthy)
# rocCurve55 <- roc(trainingData$Healthy, as.numeric(knnModel1))
# aucValue55 <- auc(rocCurve55)
# print(paste("AUC:", aucValue55))
# plot(rocCurve55, col = "blue", main = "ROC Curve for K - Nearest Neighbors (Training Data)")



cfte5 <- table(Predicted = knnModel1, Actual = testingData$Healthy)
confusionMatrix(cfte5)
mean(knnModel1 == testingData$Healthy)
#About 52% if the observation data been correctly predicted.
#The predict method youra're trying to use is not applicable to a KNN model created using the knn function from the class package.
rocCurve5 <- roc(testingData$Healthy, as.numeric(knnModel1))
aucValue5 <- auc(rocCurve5)
print(paste("AUC:", aucValue5))
plot(rocCurve5, col = "blue", main = "ROC Curve for K - Nearest Neighbors")
#K - Nearest Neighbors: Testing Data:     Accuracy: 0.5208    AUC: 0.5208333







#Ridge Regression (Classification)
library(glmnet)
#First Let's choose an optimal lamda by cross validation
ridgeCV <- cv.glmnet(as.matrix(trainingDataSubset), trainingData$Healthy, type.measure = "class", family = "binomial",
                     alpha = 0, nfolds = 10) # 10-fold cross validation
#Alpha 0 = Ridge
#Alpha 1 = Lasso
#Should I for the first param, only include predictors without the response variable??

plot(ridgeCV)
ridgeCV$lambda
ridgeCV$lambda.1se
ridgeCV$lambda.min

coef(ridgeCV, s = "lambda.min")
coef(ridgeCV, s = "lambda.1se")
#?????What does . from the coeff value represent????   it represents coefficients shrunk down to zero as a result of penalty

#Next Let's predict for the training data       #news: working on the training data
predictions6 <- predict(ridgeCV, s = ridgeCV$lambda.min, newx = as.matrix(trainingDataSubset), type = "class")
cf6 <- table(Predicted = predictions6, Actual = trainingData$Healthy)
#Let's analyze this confusion matrix in detail!
confusionMatrix(cf6)

rocCurve6 <- roc(trainingData$Healthy, as.numeric(predictions6))
aucValue6 <- auc(rocCurve6)
print(paste("AUC:", aucValue6))
plot(rocCurve6, col = "blue", main = "ROC Curve for Ridge Regression (Training Data)")
#Ridge Regression: Training Data:     Accuracy: 0.7091    AUC: 0.7082228




#Now Let's predict for the testing data
testPredictions6 <- predict(ridgeCV, s = ridgeCV$lambda.min, newx = as.matrix(testingDataSubset), type = "class")
cft6 <- table(testPredictions6, testingData$Healthy)
#Let's analyze this confusion matrix in detail!
confusionMatrix(cft6)
rocCurveT6 <- roc(testingData$Healthy, as.numeric(testPredictions6))
aucValueT6 <- auc(rocCurveT6)
print(paste("AUC:", aucValueT6))
plot(rocCurveT6, col = "blue", main = "ROC Curve for Ridge Regression")
#Ridge Regression: Testing Data:     Accuracy: 0.5833    AUC: 0.592857







#Now Let's turn into Lasso Regression (Classification) allowing some of the coefficients to shrink towards zero
lassoCV <- cv.glmnet(as.matrix(trainingDataSubset), trainingData$Healthy, type.measure = "class", family = "binomial",
                     alpha = 1, nfolds = 10) # 10-fold cross validation
# Use type.measure ="mse" for regression     family = "gaussian" used for linear regression  assumes that response var is normally distributed

plot(lassoCV)
lassoCV$lambda
lassoCV$lambda.1se
lassoCV$lambda.min

coef(lassoCV, s = "lambda.min")
coef(lassoCV, s = "lambda.1se")


#Next Let's predict for the training data       #news: working on the training data
predictions7 <- predict(lassoCV, s = lassoCV$lambda.min, newx = as.matrix(trainingDataSubset), type = "class")
cf7 <- table(predictions7, trainingData$Healthy)
#Let's analyze this confusion matrix in detail!
confusionMatrix(cf7)
rocCurve7 <- roc(trainingData$Healthy, as.numeric(predictions7))
aucValue7 <- auc(rocCurve7)
print(paste("AUC:", aucValue7))
plot(rocCurve7, col = "blue", main = "ROC Curve for Lasso Regression")
#Lasso Regression: Training Data:     Accuracy: 0.6818    AUC: 0.680371



#Now Let's predict for the testing data
testPredictions7 <- predict(lassoCV, s = lassoCV$lambda.min, newx = as.matrix(testingDataSubset), type = "class")
cft7 <- table(testPredictions7, testingData$Healthy)
#Let's analyze this confusion matrix in detail!
confusionMatrix(cft7)
rocCurveT7 <- roc(testingData$Healthy, as.numeric(testPredictions7))
aucValueT7 <- auc(rocCurveT7)
print(paste("AUC:", aucValueT7))
plot(rocCurveT7, col = "blue", main = "ROC Curve for Lasso Regression")
#Lasso Regression: Testing Data:     Accuracy: 0.5625    AUC: 0.567857






#Elastic net regression
#glmnet  param lamda    (alpha       1-alpha  )
#lamda contorls how much penalty to apply to the regression
#lamda quals 0 the whole penalty goes away   

listFitsElastic <- list()  #make an empty list first   will store the fits from the model
listFitsElastic["firrw"] = 3
listFitsElastic[5] = 3
listFitsElastic["firrw"]
a = "aa"
b = "bb"
c = "cc"
temp <- data.frame(a, b, c, D = "DSS")  #parameter passed in will be used as the name of the columns



alpha_values <- seq(0, 1, by = 0.1)
alpha_values
for(i in alpha_values){
  cv_model <- cv.glmnet(as.matrix(trainingDataSubset), trainingData$Healthy, type.measure = "class",family = "binomial",
    alpha = i, 
    nfolds = 10)
  
  listFitsElastic[[as.character(i)]] <- cv_model #if I dont sepcifiy with the bracket overwrite the previous practice code in the list.
  
}
listFitsElastic["firrw"]
listFitsElastic

#Let's choose the ideal value of alpha based on the AUC value.
predictedCVModels <- list()
aucCVModels <-list()
for(i in alpha_values)
{
  predictedCVModels[[as.character(i)]] <- predict(listFitsElastic[[as.character(i)]], s = listFitsElastic[[as.character(i)]]$lambda.min, newx = as.matrix(testingDataSubset), type = "class")
  rocCurvesCVModels <- roc(testingData$Healthy, as.numeric(predictedCVModels[[as.character(i)]]))
  aucCVModels[[as.character(i)]] <- auc(rocCurvesCVModels)
}

aucCVModels
#Based on this analysis I will choose the ideal alpha for elastic net regression to be 0.5
elasticCV <- cv.glmnet(as.matrix(trainingDataSubset), trainingData$Healthy, type.measure = "class", family = "binomial",
                     alpha = 0.5, nfolds = 10) # 10-fold cross validation

plot(elasticCV)
elasticCV$lambda
elasticCV$lambda.1se
elasticCV$lambda.min

coef(elasticCV, s = "lambda.min")
#coef(elasticCV, s = "lambda.1se")

#I chose lamda.min


#Now let's test the model with the training model
trainPredictions8 <- predict(elasticCV, s = elasticCV$lambda.min, newx = as.matrix(trainingDataSubset), type = "class")
cftr8 <- table(Predicted = trainPredictions8, Actual = trainingData$Healthy)
confusionMatrix(cftr8)
#AUC
rocCurve88 <- roc(trainingData$Healthy, as.numeric(trainPredictions8))
aucValue88 <- auc(rocCurve88)
print(paste("AUC:", aucValue88))
plot(rocCurve88, col = "blue", main = "ROC Curve for Elastic-Net Regression (Training Data")
#Elastic-Net Regression:  Training Data    Accuracy: 0.6909       AUC:0.7122

#Testing the model with the testing data
testPredictions8 <- predict(elasticCV, s = elasticCV$lambda.min, newx = as.matrix(testingDataSubset), type = "class")
cfte8 <- table(Predicted = testPredictions8, Actual = testingData$Healthy)
confusionMatrix(cfte8)
#AUC
rocCurve8 <- roc(testingData$Healthy, as.numeric(testPredictions8))
aucValue8 <- auc(rocCurve8)
print(paste("AUC:", aucValue8))
plot(rocCurve88, col = "blue", main = "ROC Curve for Elastic-Net Regression (Testing Data")
#Elastic-Net Regression:  Testing Data    Accuracy: 0.5625       AUC: 0.567857142857143





#Support Vector machines!!
library(e1071)
#Training the model SVM Support Vector Machines
svmModel <- svm(as.factor(trainingData$Healthy) ~., data = trainingDataSubset,
                type = 'C-classification',
                kernel = 'linear', #the decision boundary will be linear
                cost = 1)

#model ha used 76 support vectors which are the data points that lie closest to the decision boundary
#and are crucial in defining the postition and the orientation of the boundary
#(40, 36) 40 suppor vec from one class, 36 support vec from the other class

svmModel
summary(svmModel)

#now let's test the model
testPredictions9 <- predict(svmModel, newdata = testingDataSubset, type = "class")
cfte9 <- table(Predicted = testPredictions9, Actual = testingData$Healthy)
confusionMatrix(cfte9)

rocCurve9 <- roc(testingData$Healthy, as.numeric(testPredictions9)) 
aucValue9 <- auc(rocCurve9)
print(paste("AUC:", aucValue9))
plot(rocCurve1, col = "blue", main = "ROC Curve for Support Vector Machine")
















############################################################################################
#Now Let' do Classification Consistency!!!!

t1=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
  #Choose algorithm
  #1. Random Forest
  #2. Logistic Regression
  #3. Naive Bayes
  #4. Decision Tree (Classification)
  #5. K-Nearest Neighbiors
  #6. Ridge Regression
  #7. Lasso Regression
  
  #train the models in both data 1 and 2
  rfboot1 <- randomForest(as.factor(dat1$Healthy)~.,data = dat1Predictors,mtry = 3, ntree = 100) #fit rf in data 1
  #rfboot1 <- glm(as.factor(dat1$Healthy) ~., data = dat1Predictors, family = 'binomial')
  #rfboot1 <- naiveBayes(as.factor(dat1$Healthy) ~., data = dat1Predictors)
  #rfboot1 <- rpart(as.factor(dat1$Healthy) ~., data = dat1Predictors, method = "class", cp = 0.001) 
  #rfboot1 <- knn(train = dat1Predictors, test = testingDat1Predictors, cl = dat1$Healthy,k = round(sqrt(nrow(dat1Predictors))))
  #rfboot1 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = 0, nfolds = 10) # 10-fold cross validation
  #rfboot1 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = 1, nfolds = 10) # 10-fold cross validation
  
  rf1=predict(rfboot1,data=dat1Predictors,type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  
  
  rfboot2=randomForest(as.factor(dat2$Healthy)~.,data=dat2Predictors,mtry=4) #fit rf in data 2
  #rfboot2 <- glm(as.factor(dat1$Healthy) ~., data = dat1Predictors, family = 'binomial')
  #rfboot2 <- naiveBayes(as.factor(dat1$Healthy) ~., data = dat1Predictors)
  #rfboot2 <- rpart(as.factor(dat1$Healthy) ~., data = dat1Predictors, method = "class", cp = 0.001) 
  #rfboot2 <- knn(train = dat1Predictors, test = testingDat1Predictors, cl = dat1$Healthy,k = round(sqrt(nrow(dat1Predictors))))
  #rfboot2 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = 0, nfolds = 10) # 10-fold cross validation
  #rfboot2 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = 1, nfolds = 10) # 10-fold cross validation
  
  rf2=predict(rfboot2,data=dat2Predictors,type='class') #predicted class 
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  
  
  #rfbst1[1] indicates the threshold 1.5
  # #Step 2d: get predicted probabilities and predicted classes for testing data 
  
  #type = "class": Returns the predicted class labels.
  #type = "response": Returns the predicted probabilities (for classification) or the raw predicted values (for regression).
  rfpred1=predict(rfboot1,newdata=testingDataSubset,type='class')
  rfpred1 # is it a vector???????
  rfpred1[1] #in R index starts at 1
  
  rfpred2=predict(rfboot2,newdata=testingDataSubset,type='class')
  rfpred2
  #is.vector(rfpred1)
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t1[[i]] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  example
  #Code Trial: for (j in 1:testingData)  rfpredConsistency1 <- ifelse(rfpred1[j] == rfpred2[j], 1, 0)
  
  #First trial Result:
  #Consistency???? 14 data points were inconsistent out of 48
  #Inconsistency rate: 0.2916667
  #Consistency Rate: 0.7083333
  
}   #10 iterations

for(i in 1:1000){
  print(paste("Random Forest Consistency rate[",i, "]: ",t1[[i]]))
}
#vecc <- do.call(c,t1)  //make sure to use rbind!!!
#mean(t1)
#t1[1]
summary(do.call(rbind,t1)) #five number summary of classification agreement


write.csv(summary(do.call(rbind,t1)),"Consistency.RandomForest.csv")
#do call funbction: allows you to call other functions using a list of arguments
#particulary useful for when we have a function and its arguments stored in separate objects (e.g., in a list) and we want to apply the function to these arguments.
#do.call(c, t1) effectively concatenates the elements of t1 into a single vector.
#summary() summary function, you need to pass in a numerical vector

#standard deviation of classification agreement estimates
sd1 = sd(do.call(rbind,t1))
print(paste("Random Forest Classification Standard Deviation: ", sd1))

#Let's draw a histogram for Random Forest Classification
pdf("Fig.1.Cons.RF.pdf")
hist(do.call(rbind,t1),main='Classification Agreement Estimates \nfrom Random Forest',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()
#hist might pass in a numeri vector
#legend(x = .86, y = 3.1, legend = "Trials: 10", bg = "yellow",
#      cex = 0.7)
#legend("topright", legend = c("Data 1", "Data 2"), fill = c("blue", "red"))











#Quesition?????
hist(c(1,2,3,4,5))



###################################################################################################
#Consistency from Logistic Regression
t2=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
  
  #train the models in both data 1 and 2
  rfboot1 <- glm(as.factor(dat1$Healthy) ~., data = dat1Predictors, family = 'binomial')
  #rfboot1 <- stepAIC(rfboot1)
  rf1=predict(rfboot1,data=dat1Predictors,type='response') #predicted class
  rf1 <- ifelse(rf1 >0.5, 1, 0) #Needed!! for classification for logistic Regression
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- glm(as.factor(dat2$Healthy) ~., data = dat2Predictors, family = 'binomial')
  #rfboot2 <- stepAIC(rfboot2)
  rf2=predict(rfboot2,data=dat2Predictors,type='response') #predicted class
  rf2 <- ifelse(rf2 >0.5, 1, 0) #Needed!! for classification for logistic Regression
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2

  rfpred1=predict(rfboot1,newdata=testingDataSubset,type='response')
  rfpred1 <- ifelse(rfpred1 >0.5, 1, 0) #Needed!! for classification for logistic Regression
  rfpred1
  rfpred1[1]
  
  rfpred2=predict(rfboot2,newdata=testingDataSubset,type='response')
  rfpred2 <- ifelse(rfpred2 >0.5, 1, 0) #Needed!! for classification for logistic Regression
  rfpred2
  #is.vector(rfpred1)
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t2[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t2[i]
  
}

for(i in 1:1000){
  print(paste("Logistic Regression Consistency rate[",i, "]: ",t2[i]))
}
#do.call(rbind,t2) #through rbind  I can turn into a vectorrr!!!!!!!!!!!!!!!!!!!!1
#Step 3: aggregate and report results
summary(do.call(rbind,t2)) #five number summary of classification agreement
sd2 = sd(do.call(rbind,t2))
print(paste("Logistic Regression Classification Standard Deviation: ", sd2))

pdf("Fig.2.Cons.LR.pdf")
hist(do.call(rbind,t2),main='Classification Agreement Estimates \nfrom Logistic Regression(ALC)',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()


#we can put lists can nhave multiple data types.   in array
#building is an list each floor object 
#matrix  it has to be number 
#dataframe: if has text variables 
#array can handle multiple dimension

#if treating each var as adimension  then in matrix we colapsing them int  2d imendional array special case
#vector is a one dimensional matrix

#using matrix, calculations.   bring row s and columns together.

#each dimension is a model.extract()
#regression is only a 2 dimensional
#got every regression line for every trees
#pca   each component  is a equation. 





###################################################################################################
#Consistency from Naive Bayes
t3=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
  
  #train the models in both data 1 and 2
  rfboot1 <- naiveBayes(as.factor(dat1$Healthy) ~., data = dat1Predictors)
  rf1=predict(rfboot1, newdata=dat1Predictors,type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- naiveBayes(as.factor(dat2$Healthy) ~., data = dat2Predictors)
  rf2=predict(rfboot2, newdata=dat2Predictors,type='class') #predicted class
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  rfpred1=predict(rfboot1,newdata=testingDataSubset,type='class')
  rfpred1
  rfpred1[1]
  
  rfpred2=predict(rfboot2,newdata=testingDataSubset,type='class')
  rfpred2
  #is.vector(rfpred1)
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t3[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t3[i]
  
}

for(i in 1:1000){
  print(paste("Naive Bayes Consistency rate[",i, "]: ",t3[i]))
}

#Step 3: aggregate and report results
summary(do.call(rbind,t3)) #five number summary of classification agreement
sd3 = sd(do.call(rbind,t3))
print(paste("Naive Bayes Classification Standard Deviation: ", sd3))
pdf("Fig.3.Cons.NaiveBayes.pdf")
hist(do.call(rbind,t3),main='Classification Agreement Estimates \nfrom Naive Bayes',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()









###################################################################################################
#Consistency from Decision Tree
t4=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
  
  #train the models in both data 1 and 2
  rfboot1 <- rpart(as.factor(dat1$Healthy) ~., data = dat1Predictors, method = "class", cp = 0.001) 
  rf1=predict(rfboot1, newdata=dat1Predictors,type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- rpart(as.factor(dat2$Healthy) ~., data = dat2Predictors, method = "class", cp = 0.001) 
  rf2=predict(rfboot2, newdata=dat2Predictors,type='class') #predicted class
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  rfpred1=predict(rfboot1,newdata=testingDataSubset,type='class')
  rfpred1
  rfpred1[1]
  
  rfpred2=predict(rfboot2,newdata=testingDataSubset,type='class')
  rfpred2
  #is.vector(rfpred1)
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t4[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t4[i]
  
}

for(i in 1:1000){
  print(paste("Decision Tree Consistency rate[",i, "]: ",t4[i]))
}

#Step 3: aggregate and report results
summary(do.call(rbind,t4)) #five number summary of classification agreement
sd4 = sd(do.call(rbind,t4))
print(paste("Decision Tree Classification Standard Deviation: ", sd4))
pdf("Fig.4.Cons.DecisionTree.pdf")
hist(do.call(rbind,t4),main='Classification Agreement Estimates \nfrom Decision Tree',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()











###################################################################################################
#Consistency from K-Nearest Neighbors
t5=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
 
  #train the models in both data 1 and 2
  rfboot1 <- knn(train = dat1Predictors, test = testDat1Predictors, cl = dat1$Healthy,k = round(sqrt(nrow(dat1Predictors))))
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot1 <- knn(train = dat2Predictors, test = testDat2Predictors, cl = dat2$Healthy,k = round(sqrt(nrow(dat2Predictors))))
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  #Knn function doesn't need predict functions
  rfpred1
  rfpred2
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t5[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t5[i]
  
}

for(i in 1:1000){
  print(paste("K-Nearest Neighbors Consistency rate[",i, "]: ",t5[i]))
}

#Step 3: aggregate and report results
summary(do.call(rbind,t5)) #five number summary of classification agreement
sd5 = sd(do.call(rbind,t5))
print(paste("K-Nearest Neighbors Classification Standard Deviation: ", sd5))
pdf("Fig.5.Cons.KNN.pdf")
hist(do.call(rbind,t5),main='Classification Agreement Estimates \nfrom K-Nearest Neighbors',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()
####Question!!!









###################################################################################################
#Consistency from Ridge Regression
t6=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  tryCatch(
    {   # use this function to skip to next iteration/condition if errors
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
  
  #train the models in both data 1 and 2
  rfboot1 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = 0, nfolds = 10) # 10-fold cross validation
  rfboot1$lambda
  rfboot1$lambda.1se
  rfboot1$lambda.min
  rf1=predict(rfboot1, s = rfboot1$lambda.min, newx=as.matrix(dat1Predictors),type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- cv.glmnet(as.matrix(dat2Predictors), dat2$Healthy, type.measure = "class", family = "binomial", alpha = 0, nfolds = 10) # 10-fold cross validation
  rfboot2$lambda
  rfboot2$lambda.1se
  rfboot2$lambda.min
  rf2=predict(rfboot2, s = rfboot2$lambda.min, newx=as.matrix(dat2Predictors),type='class') #predicted class
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  
  rfpred1=predict(rfboot1,s = rfboot1$lambda.min, newx=as.matrix(testingDataSubset),type='class')
  rfpred1
  
  rfpred2=predict(rfboot1,s = rfboot2$lambda.min, newx=as.matrix(testingDataSubset),type='class')
  rfpred2
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t6[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t6[i]
  
    }, error = function(err) {print('Error')
      return(NA)
    }) # this is part of the Trycatch function that skips if error.
}

for(i in 1:1000){
  print(paste("Ridge Regression Consistency rate[",i, "]: ",t6[i]))
}

#Step 3: aggregate and report results
summary(do.call(rbind,t6)) #five number summary of classification agreement
sd6 = sd(do.call(rbind,t6))
print(paste("Ridge Regression Classification Standard Deviation: ", sd6))
pdf("Fig.6.Cons.Ridge.pdf")
hist(do.call(rbind,t6),main='Classification Agreement Estimates \nfrom Ridge Regression',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()







###################################################################################################
#Consistency from Lasso Regression
t7=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  tryCatch(
    {   # use this function to skip to next iteration/condition if errors
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  #########################################################################################
  #Havent used this portion of the code
  testBoot1=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testBoot2=sample(1:nrow(testingData),nrow(testingData),replace=TRUE) 
  testDat1=testingData[testBoot1,1:20] 
  testDat2=testingData[testBoot2,1:20]
  
  testDat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  testDat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  ##########################################################################################3
  
  
  #train the models in both data 1 and 2
  rfboot1 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = 1, nfolds = 10) # 10-fold cross validation
  rfboot1$lambda
  rfboot1$lambda.1se
  rfboot1$lambda.min
  rf1=predict(rfboot1, s = rfboot1$lambda.min, newx=as.matrix(dat1Predictors),type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- cv.glmnet(as.matrix(dat2Predictors), dat2$Healthy, type.measure = "class", family = "binomial", alpha = 1, nfolds = 10) # 10-fold cross validation
  rfboot2$lambda
  rfboot2$lambda.1se
  rfboot2$lambda.min
  rf2=predict(rfboot2, s = rfboot2$lambda.min, newx=as.matrix(dat2Predictors),type='class') #predicted class
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  
  rfpred1=predict(rfboot1,s = rfboot1$lambda.min, newx=as.matrix(testingDataSubset),type='class')
  rfpred1
  
  rfpred2=predict(rfboot1,s = rfboot2$lambda.min, newx=as.matrix(testingDataSubset),type='class')
  rfpred2
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t7[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t7[i]
  
    }, error = function(err) {print('Error')
      return(NA)
    }) # this is part of the Trycatch function that skips if error.
}

for(i in 1:1000){
  print(paste("Lasso Regression Consistency rate[",i, "]: ",t7[i]))
}

#Step 3: aggregate and report results
summary(do.call(rbind,t7)) #five number summary of classification agreement
sd7 = sd(do.call(rbind,t7))
print(paste("Lasso Regression Classification Standard Deviation: ", sd7))
pdf("Fig.7.Cons.LASSO.pdf")
hist(do.call(rbind,t7),main='Classification Agreement Estimates \nfrom Lasso Regression',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()









###################################################################################################
#Consistency from Elastic-Net Regression

elasticTempResults <- data.frame()
alpha_values
for(j in alpha_values)
{
  t8=list(NULL) #to save classification results in an array.
  
  for (i in 1:1000){
    tryCatch(
      {   # use this function to skip to next iteration/condition if errors
    #bootstrapping the data!
    boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
    boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
    dat1=trainingData[boot1,1:20] 
    dat2=trainingData[boot2,1:20]
    
    #for training the model using the training data
    dat1Predictors <- dat1 %>%
      dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
    
    dat2Predictors <- dat2 %>%
      dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
    
    
    
    #train the models in both data 1 and 2
    rfboot1 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, type.measure = "class", family = "binomial", alpha = j, nfolds = 10) # 10-fold cross validation
    rfboot1$lambda
    rfboot1$lambda.1se
    rfboot1$lambda.min
    rf1=predict(rfboot1, s = rfboot1$lambda.min, newx=as.matrix(dat1Predictors),type='class') #predicted class
    rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
    rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
    #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
    rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
    #flattens the list or data frame into a vector.
    rfbst1
    rf1
    
    rfboot2 <- cv.glmnet(as.matrix(dat2Predictors), dat2$Healthy, type.measure = "class", family = "binomial", alpha = j, nfolds = 10) # 10-fold cross validation
    rfboot2$lambda
    rfboot2$lambda.1se
    rfboot2$lambda.min
    rf2=predict(rfboot2, s = rfboot2$lambda.min, newx=as.matrix(dat2Predictors),type='class') #predicted class
    rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
    rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
    rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
    rfbst2
    rf2
    
    
    rfpred1=predict(rfboot1,s = rfboot1$lambda.min, newx=as.matrix(testingDataSubset),type='class')
    rfpred1
    
    rfpred2=predict(rfboot1,s = rfboot2$lambda.min, newx=as.matrix(testingDataSubset),type='class')
    rfpred2
    
    rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
    confusionMatrix(rfpredConsistency1)
    #to extract the accuracy rate from the confusion table
    t8[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
    t8[i]
      }, error = function(err) {print('Error')
        return(NA)
      }) # this is part of the Trycatch function that skips if error.
  }
  
  Median <- summary(do.call(rbind,t8))
  Median <- Median[3,1]
  median_value <- as.numeric(sub(".*: *", "", Median))
  median_value
  
  Mean <- summary(do.call(rbind,t8))
  Mean <- Mean[4,1]
  mean_value <- as.numeric(sub(".*: *", "", Mean))
  mean_value
  
  
  #tempRow <- data.frame(Alpha = j, Median = summary(do.call(rbind,t8))["Median"], Mean = summary(do.call(rbind,t8))["Mean"]) #five number summary of classification agreement
  tempRow <- data.frame(Alpha = j, Median = median_value, Mean = mean_value) #five number summary of classification agreement
  elasticTempResults <- rbind(elasticTempResults, tempRow)
  
}
  write.csv(elasticTempResults, "elasticTempResults.csv")
  elasticTempResults
  alpha.m<-elasticTempResults[which(elasticTempResults$Mean==max(elasticTempResults$Mean)),1]
  
#Now let's choose the best alpha value for classfication

t8=list(NULL) #to save classification results in an array.

for (i in 1:1000){
  tryCatch(
    {   # use this function to skip to next iteration/condition if errors
      #bootstrapping the data!
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  
  
  #train the models in both data 1 and 2
  rfboot1 <- cv.glmnet(as.matrix(dat1Predictors), dat1$Healthy, 
                       type.measure = "class", family = "binomial", 
                       alpha = alpha.m, nfolds = 10) # 10-fold cross validation
  rfboot1$lambda
  rfboot1$lambda.1se
  rfboot1$lambda.min
  rf1=predict(rfboot1, s = rfboot1$lambda.min, newx=as.matrix(dat1Predictors),type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- cv.glmnet(as.matrix(dat2Predictors), dat2$Healthy, 
                       type.measure = "class", family = "binomial", 
                       alpha = alpha.m, nfolds = 10) # 10-fold cross validation
  rfboot2$lambda
  rfboot2$lambda.1se
  rfboot2$lambda.min
  rf2=predict(rfboot2, s = rfboot2$lambda.min, newx=as.matrix(dat2Predictors),type='class') #predicted class
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  
  rfpred1=predict(rfboot1,s = rfboot1$lambda.min, newx=as.matrix(testingDataSubset),type='class')
  rfpred1
  
  rfpred2=predict(rfboot1,s = rfboot2$lambda.min, newx=as.matrix(testingDataSubset),type='class')
  rfpred2
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t8[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t8[i]

    }, error = function(err) {print('Error')
      return(NA)
    }) # this is part of the Trycatch function that skips if error.
}


for(i in 1:1000){
  print(paste("Elastic-Net Regression Consistency rate[",i, "]: ",t8[i]))
}

#Step 3: aggregate and report results
summary(do.call(rbind,t8)) #five number summary of classification agreement
sd8 = sd(do.call(rbind,t8))

print(paste("Elastic-Net Regression Standard Deviation: ", sd8))
pdf("Fig.8.Cons.ElasticNet.pdf")
hist(do.call(rbind,t8),main='Classification Agreement Estimates \nfrom Ridge Elastic-Net Regression',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()





################################################################################################
#Consistency from Support Vector Machine
t9=list(NULL) #to save classification results in an array.
for (i in 1:1000){
  #bootstrapping the data!
  boot1=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  boot2=sample(1:nrow(trainingData),nrow(trainingData),replace=TRUE) 
  dat1=trainingData[boot1,1:20] 
  dat2=trainingData[boot2,1:20]
  
  #for training the model using the training data
  dat1Predictors <- dat1 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  
  dat2Predictors <- dat2 %>%
    dplyr::select(-Participant_ID, -Diagnosis, -Healthy, -SCC, -MCI, -Wave, -Study, -Use, -ComplianceRate)
  

  
  
  #train the models in both data 1 and 2
  rfboot1 <- svm(as.factor(dat1$Healthy) ~., data = dat1Predictors,
                 type = 'C-classification',
                 kernel = 'linear', #the decision boundary will be linear
                 cost = 1)
  rf1=predict(rfboot1,data=dat1Predictors,type='class') #predicted class
  rfroc1=roc(dat1$Healthy, as.numeric(rf1)) #ROC curve 
  rfbst1=coords(rfroc1,'best')[1] #Youden cutpoint
  #extracts specific coordinates from the ROC curve. The argument 'best' typically indicates that you are looking for the point on the ROC curve that maximizes the Youden index
  rfbst1=unlist(rfbst1)[1] #Youden cutpoint 
  #flattens the list or data frame into a vector.
  rfbst1
  rf1
  
  rfboot2 <- svm(as.factor(dat2$Healthy) ~., data = dat2Predictors,
                 type = 'C-classification',
                 kernel = 'linear', #the decision bounary will be linear
                 cost = 1)
  rf2=predict(rfboot2,data=dat2Predictors,type='class') #predicted class
  rfroc2=roc(dat2$Healthy, as.numeric(rf2)) #ROC curve 
  rfbst2=coords(rfroc2,'best')[1] #Youden cutpoint 
  rfbst2=unlist(rfbst2)[1] #Youden cutpoint 
  rfbst2
  rf2
  
  rfpred1=predict(rfboot1,newdata=testingDataSubset,type='class')
  rfpred1
  rfpred1[1]
  
  rfpred2=predict(rfboot2,newdata=testingDataSubset,type='class')
  rfpred2
  #is.vector(rfpred1)
  
  rfpredConsistency1 <- table(Predicted = rfpred1, Actual = rfpred2)
  confusionMatrix(rfpredConsistency1)
  #to extract the accuracy rate from the confusion table
  t9[i] = sum(diag(table(rfpred1,rfpred2)))/nrow(testingData)
  t9[i]
  
}

for(i in 1:1000){
  print(paste("Support Vector Machine(SVM): Consistency rate[",i, "]: ",t9[i]))
}
#do.call(rbind,t2) #through rbind  I can turn into a vectorrr!!!!!!!!!!!!!!!!!!!!1
#Step 3: aggregate and report results
summary(do.call(rbind,t9)) #five number summary of classification agreement
sd9 = sd(do.call(rbind,t9))
print(paste("Support Vector Machine Classification Standard Deviation: ", sd9))
pdf("Fig.9.Cons.SVM.pdf")
hist(do.call(rbind,t9),main='Classification Agreement Estimates \nfrom Support Vector Machine (SVM)',xlab='Classification Agreement',ylab= 'Frequency (Consistency)',col = blues9)
dev.off()











################################################################################################
##Now let's rank the predictors!! using PCA

pca <- prcomp(as.matrix(trainingDataSubset), scale = TRUE)
plot(pca$x[,1], pca$x[,2]) #to draw 2dplot that uses the first two pcs
#x contains  principal components to draw a graph   pc1 accounts for the most varaition in the original data

pcaVariation <- pca$sdev^2  #to compute variation for each Principal component
pcaVariation
pcaVariationPercentage <- round(pcaVariation/sum(pcaVariation)*100, 1)
pcaVariationPercentage
barplot(pcaVariationPercentage, main = "PCA Scree Plot", xlab = "Principal Component (Predictors)",
        ylab = "Percent Variation")


library(ggplot2)

pca.data <- data.frame(Sample=rownames(pca$x),
                       X=pca$x[,1],
                       Y=pca$x[,2])
pca.data

ggplot(data=pca.data, aes(x=X, y=Y, label=Sample)) +
  geom_text() +
  xlab(paste("PC1 - ", pcaVariationPercentage[1], "%", sep="")) +
  ylab(paste("PC2 - ", pcaVariationPercentage[2], "%", sep="")) +
  theme_bw() +
  ggtitle("My PCA Graph")



loadingScores <- pca$rotation[,1]
loadingScores
predictorScores <- abs(loadingScores)
predictorScores
predictorScoresRanked <- sort(predictorScores, decreasing = TRUE)
predictorScoresRanked

predictorRank <- names(predictorScoresRanked[1:11])
predictorRank

pca$rotation[predictorRank,1]  #to show the scores
pca$rotation




library(factoextra)
library(rstatix)
#Scree plot of variance
fviz_eig(pca, ylim = c(0, 30))

fviz_pca_biplot(pca)
fviz_pca_biplot(pca, label = "var") #To hide data point labels

#Biplot with coloring
fviz_pca_biplot(pca, label = "var", habillage = trainingData$Healthy) #To hide data point labels
fviz_pca_biplot(pca, label = "var", habillage = trainingData$Healthy, col.var = "black") #To change the color of the vector

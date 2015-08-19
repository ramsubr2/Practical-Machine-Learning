library(caret)
library(AppliedPredictiveModeling)
library(e1071)
data(AlzheimerDisease)

# Q1. create training and test sets with about 50% of the observations 
# assigned to each

adData <- data.frame(diagnosis,predictors)
trainIndex = createDataPartition(diagnosis, p = 0.50,list=FALSE)
training = adData[trainIndex,]
testing = adData[-trainIndex,]
# this is option C

# Q2. Make a histogram and confirm the SuperPlasticizer variable is skewed. Normally you might use the log transform
# to try to make the data more symmetric. Why would that be a poor choice for this variable?

data(concrete)
set.seed(1000)
inTrain = createDataPartition(mixtures$CompressiveStrength, p = 3/4)[[1]]
training = mixtures[ inTrain,]
testing = mixtures[-inTrain,]

hist(mixtures$Superplasticizer) # histogram is right skewed
# If you View(mixtures), you see that superPlasticizer has a number of zero values
# taking the log of this is -Inf, which is not a good idea for this variable.

# Q3. Find all the predictor variables in the training set that begin with IL. 
# Perform principal components on these variables with the preProcess() function from the caret package. 
# Calculate the number of principal components needed to capture 80% of the variance

set.seed(3433)
data(AlzheimerDisease)
adData = data.frame(diagnosis,predictors)
inTrain = createDataPartition(adData$diagnosis, p = 3/4)[[1]]
training = adData[ inTrain,]
testing = adData[-inTrain,]
#Separate predictor variables with IL for training data
pv_with_IL_training <- training[,grep('^IL', x = names(training) )]
trans_pca = preProcess(pv_with_IL_training,
                   method=c("BoxCox", "center","scale", "pca"),
                   thresh=0.8, outcome=training$diagnosis)
#trans_pca  - will say that PCA needed 7 components to capture 80% of the variance

#Q4. compare accuracy of PCA vs non-PCA
# PCA

trainModel <- predict(trans_pca, pv_with_IL_training)
modelFit <- train(training$diagnosis ~ ., method="glm", data=trainModel)
#Separate predictor variables with IL for testing data
pv_with_IL_testing <- testing[,grep('^IL', x = names(testing))]
testModel <- predict(trans_pca,pv_with_IL_testing)
confusionMatrix(testing$diagnosis,predict(modelFit,testModel))
#Accuracy = 0.92

#non PCA
trans_non_pca = preProcess(pv_with_IL_training,
                       method=c("center","scale"))
trainModel_non_pca <- predict(trans_non_pca, pv_with_IL_training)
modelFit_non_pca <- train(training$diagnosis ~ ., method = "glm", data = trainModel_non_pca)

testModel_non_pca <- predict(trans_non_pca,pv_with_IL_testing)
confusionMatrix(testing$diagnosis,predict(modelFit_non_pca,testModel_non_pca))
#Accuracy = 0.65
                



#### APM - Case Study: Job Scheduling in High Performance Computing Environments ####

## The data is contained in the AppliedPredictiveModeling Package.
library(AppliedPredictiveModeling)
library(caret)
library(rpart)

data(schedulingData)

set.seed(1104)
inTrain <- createDataPartition(schedulingData$Class,
                               p = 0.8,
                               list = FALSE)

schedulingData$NumPending <- schedulingData$NumPending + 1
trainData <- schedulingData[inTrain, ]
testData <- schedulingData[-inTrain, ]

## A custom cost function is created to judge the models:
cost <- function(pred, obs) {
    isNA <- is.na(pred)
    if(!all(isNA)) {
        pred <- pred[!isNA]
        obs <- obs[!isNA]
        
        cost <- ifelse(pred == obs, 0, 1)
        if(any(pred == "VF" & obs == "L"))
            cost[pred == "VF" & obs == "L"] <- 10
        if(any(pred == "F" & obs == "L"))
            cost[pred == "F" & obs == "L"] <- 5
        if(any(pred == "F" & obs == "M"))
            cost[pred == "F" & obs == "M"] <- 5
        if(any(pred == "VF" & obs == "M"))
            cost[pred == "VF" & obs == "M"] <- 5
        
        out <- mean(cost)
    } else out <- NA
    out
}

costSummary <- function(data, lev = NULL, model = NULL) {
    if(is.character(data$obs)) 
        data$obs <- factor(data$obs, levels = lev)
    c(postResample(data[, "pred"], data[, "obs"]),
      Cost = cost(data[, "pred"], data[, "obs"]))
}

ctrl <- trainControl(method = "repeatedcv", repeats = 5,
                     summaryFunction = costSummary)


## For the cost-sensitive tree models, a matrix representation of the costs was 
## also created.

costMatrix <- ifelse(diag(4) == 1, 0, 1)
costMatrix[1, 4] <- 10
costMatrix[1, 3] <- 5
costMatrix[2, 4] <- 5
costMatrix[2, 3] <- 5
rownames(costMatrix) <- levels(trainData$Class)
colnames(costMatrix) <- levels(trainData$Class)
costMatrix


## The tree based methods did not use independent categories, but the other models 
## require that the categorical predictors (e.g. protocol) are decomposed into dummy
## variables. A model formula was created that log transforms severl of the predictors.
modForm <- as.formula(Class ~ Protocol + log10(Compounds) + log10(InputFields) +
                          log10(Iterations) + log10(NumPending) + Hour + Day)


### Cost Sensitive CART Model:
set.seed(857)
predictors <- names(trainData)[names(trainData) != "Class"]

rpFitCost <- train(x = trainData[, predictors],
                   y = trainData$Class,
                   method = "rpart",
                   metric = "Cost",
                   maximize = FALSE,
                   tuneLength = 20,
                   parms = list(loss = t(costMatrix)),
                   trControl = ctrl)

## rpart structures the cost matrix so that the true class are i rows, hence we
## transposed the costMatrix above.

#### Cost Sensitive C5.0 Model:
set.seed(857)
c50Cost <- train(x = trainData[, predictors],
                 y = trainData$Class,
                 method = "C5.0",
                 metric = "Cost",
                 maximize = FALSE,
                 costs = costMatrix,
                 tuneGrid = expand.grid(.trials = c(1, (1:10)*10),
                                        .model = "tree",
                                        .window = c(TRUE, FALSE)),
                 trControl = ctrl)


#### Weighted SVM:
set.seed(857)
svmRFitCost <- train(modForm, data = trainData, method = "svmRadial",
                     metric = "Cost",
                     maximize = FALSE,
                     preProc = c("center", "scale"),
                     class.weights = c(VF = 1, F = 1, M = 5, L = 10),
                     tuneLength = 15,
                     trControl = ctrl)

## The resampled version of confusion matrices were computed using the 
## confusionMatrix() function on the objects produced by train function:

confusionMatrix(rpFitCost, norm = "none")

## The norm argument determines how the raw counts from each resample should be
## normalized. A value of "none" results in the average counts in each cell of 
## the table. Using norm = "overall" first divides the cell entries by the total
## number of data points in the table and then averages these percentages.


################################################################################
### Session Information

sessionInfo()
rm(list = ls())
q("no")
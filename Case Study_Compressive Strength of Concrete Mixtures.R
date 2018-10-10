#### APM - Case Study: Comprehensive Strength of Concrete Mixtures ####

##install.packages(c("Hmisc", "desirability"))

library(AppliedPredictiveModeling)
library(Hmisc)
library(caret)
library(plyr)
library(desirability)

data("concrete")  ##Contains data sets concrete & mixtures
str(concrete)   ##Contains original data in amounts.
str(mixtures)  ##Contains modified data in proportions.

?describe
describe(mixtures)   
##describe() Function belongs to Hmisc package. 

##To create the scatter plot of response vs predictors, use caret function featurePlot().
featurePlot(concrete[, -9], concrete$CompressiveStrength,
            between = list(x = 1, y = 1),
            type = c("g", "p", "smooth"))

##Note that several ingredients have a large frequency for a single value & that 
##compressive strength vary widely for that value. This indicates that some of 
##the partitioning methods such as tree or MARS models may perform better. A tree
##or rule based models can models such sub groups.


##There are 19 replicate mistures with replicate data points. i.e. same value for
##all predictor variables. The replicate mixture cannot be treated as different
##samples(bcoz test-train split) and hence were average to unique values.
##We use the ddply() function from plyr package for this purpose.
?ddply
averaged = ddply(mixtures, .(Cement, BlastFurnaceSlag, FlyAsh, Water, 
                             Superplasticizer, CoarseAggregate, FineAggregate, 
                             Age), function(x) c(CompressiveStrength = 
                                                   mean(x$CompressiveStrength)))



#### Model building Strategy: ####

##A random held out set of 25% (n = 247) will be used 
##as test set and 5 repeats of 10 fold CV will be used to tune various models.
##A number of models will be created and evaluated. Once a final model is selected, the
##model is used to predict mixture with optimal compressive strength within 
##practical limitations.

##As proportion of each component is used as predictors, there is a built-in 
##dependency in the predictor values (bcoz x = 1-remaining_all). Despite this 
##pairwise correlations are not large and therefore we would not expect methods
##are designed to deal with collinearity to have superior performance compare to
##other models.

set.seed(975)
inTrain = createDataPartition(averaged$CompressiveStrength, p = 3/4)[[1]]
##CreateDataPartition() creates a number (Default =1) of training - test splits.
training = averaged[inTrain, ]
testing = averaged[-inTrain, ] ##Held out set.

ctrl = trainControl(method = "repeatedcv", repeats = 5, number = 10)


#### Linear Regression: ####

##We build a linear regression model with all predictors, their squared terms, 
##and all second order interactions. We first create a formula object using 
##paste() & as.formula() and use the same.

modFormula = paste("CompressiveStrength ~ (.)^2 + I(Cement^2) + ",
                   "I(BlastFurnaceSlag^2) + I(FlyAsh^2) + I(Water^2) + ",
                   "I(Superplasticizer^2) + I(CoarseAggregate^2) + ",
                   "I(FineAggregate^2) + I(Age^2)")

##The "(.)^2" expands into a model with all predictors and their two-factor 
##interactions.

modFormula = as.formula(modFormula)

set.seed(669)
lmFit = train(modFormula, data = training,
              method = "lm",
              trControl = ctrl)

lmFit
lmFit$finalModel ##Model Coefficients.


#### PLS Regression: ####
set.seed(669)
plsFit <- train(modFormula, data = training,
                method = "pls",
                preProc = c("center", "scale"),
                tuneLength = 15,
                trControl = ctrl)

plsFit
names(plsFit)
plsFit$finalModel

#### ENet (LASSO) Regression: ####

lassoGrid <- expand.grid(lambda = c(0, .001, .01, .1), 
                         fraction = seq(0.05, 1, length = 20))

set.seed(669)
lassoFit <- train(modFormula, data = training,
                  method = "enet",
                  preProc = c("center", "scale"),
                  tuneGrid = lassoGrid,
                  trControl = ctrl)


#### MARS Regression: ####
set.seed(669)
earthFit <- train(CompressiveStrength ~ ., data = training,
                  method = "earth",
                  tuneGrid = expand.grid(degree = 1, 
                                         nprune = 2:25),
                  trControl = ctrl)


#### SVM Regression: ####
set.seed(669)
svmRFit <- train(CompressiveStrength ~ ., data = training,
                 method = "svmRadial",
                 tuneLength = 15,
                 preProc = c("center", "scale"),
                 trControl = ctrl)

#### Neural Network Regression: ####
nnetGrid <- expand.grid(decay = c(0.001, .01, .1), 
                        size = seq(1, 27, by = 2), 
                        bag = FALSE)

set.seed(669)
nnetFit <- train(CompressiveStrength ~ .,
                 data = training,
                 method = "avNNet",
                 tuneGrid = nnetGrid,
                 preProc = c("center", "scale"),
                 linout = TRUE,
                 trace = FALSE,
                 maxit = 1000,
                 allowParallel = FALSE,
                 trControl = ctrl)


#### Regression & Model Trees: ####
set.seed(669)
rpartFit <- train(CompressiveStrength ~ .,
                  data = training,
                  method = "rpart",
                  tuneLength = 30,
                  trControl = ctrl)


set.seed(669)
treebagFit <- train(CompressiveStrength ~ .,
                    data = training,
                    method = "treebag",
                    trControl = ctrl)

set.seed(669)
ctreeFit <- train(CompressiveStrength ~ .,
                  data = training,
                  method = "ctree",
                  tuneLength = 10,
                  trControl = ctrl)

#### Random Forest ####
set.seed(669)
rfFit <- train(CompressiveStrength ~ .,
               data = training,
               method = "rf",
               tuneLength = 10,
               ntrees = 1000,
               importance = TRUE,
               trControl = ctrl)


#### Boosting ####
gbmGrid <- expand.grid(interaction.depth = seq(1, 7, by = 2),
                       n.trees = seq(100, 1000, by = 50),
                       shrinkage = c(0.01, 0.1))
set.seed(669)
gbmFit <- train(CompressiveStrength ~ .,
                data = training,
                method = "gbm",
                tuneGrid = gbmGrid,
                verbose = FALSE,
                trControl = ctrl)


#### Cubist ####
cbGrid <- expand.grid(committees = c(1, 5, 10, 50, 75, 100), 
                      neighbors = c(0, 1, 3, 5, 7, 9))
set.seed(669)
cbFit <- train(CompressiveStrength ~ .,
               data = training,
               method = "cubist",
               tuneGrid = cbGrid,
               trControl = ctrl)

#### Model Trees ####
set.seed(669)
mtFit <- train(CompressiveStrength ~ .,
               data = training,
               method = "M5",
               trControl = ctrl)


################################################################################

####### Model Performance #######

## Collect the resampling statistics across all the models into a single Object.
## Use the caret function resamples() for the same:

rs <- resamples(list("Linear Reg" = lmFit, 
                     "PLS" = plsFit,
                     "Elastic Net" = lassoFit, 
                     "MARS" = earthFit,
                     "SVM" = svmRFit, 
                     "Neural Networks" = nnetFit,
                     "CART" = rpartFit, 
                     "Cond Inf Tree" = ctreeFit,
                     "Bagged Tree" = treebagFit,
                     "Boosted Tree" = gbmFit,
                     "Random Forest" = rfFit,
                     "Cubist" = cbFit))


parallelplot(rs) ##Using RMSE.
parallelplot(rs, metric = "Rsquared") ##Using R-squared.

##Other visualizations of resampling results can also be created. 
##See ?xyplot.resamples for details.

## Get the test set results across several models
nnetPred <- predict(nnetFit, testing)
gbmPred <- predict(gbmFit, testing)
cbPred <- predict(cbFit, testing)

testResults <- rbind(postResample(nnetPred, testing$CompressiveStrength),
                     postResample(gbmPred, testing$CompressiveStrength),
                     postResample(cbPred, testing$CompressiveStrength))
testResults <- as.data.frame(testResults)
testResults$Model <- c("Neural Networks", "Boosted Tree", "Cubist")
testResults <- testResults[order(testResults$RMSE),]


################################################################################
### Optimizing Compressive Strength

## We first use the 28-day data to generate a set of random starting 
##points from the training set. The data are center-scaled as the distance 
##between the formulation are used as a measure of dissimilarity for maximum 
##dissimilarity sampling.

age28Data = subset(trainingData, Age==28)
##Now we remove the age and the compressive strength column and then center-scale
##the predictors columns.

pp1 = preProcess(age28Data[, -(8:9)], c("center", "scale"))
scaledTrain = predict(pp1, age28Data[, 1:7])
set.seed(91)
startMixture = sample(1:nrow(age28Data), 1)
starters = scaledTrain[startMixture, 1:7]

##After this, maximum dissimilarity sampling method selects 14 more mixtures to 
##complete a diverse set of starting points for the search algorithm.
pool = scaledTrain
index = maxDissim(starters, pool, 14)
startPoints = c(startMixture, index)
starters = age28Data[startPoints, 1:7]

##Since all proportions will add up to one, we remove the proportion for water
##water proportion can be computed using 1 - sum_of_others. Without this 
##step, the search procedure would pick candidate mixture values that would not
##add to one.

startingValues = starters[, -4]

##To maximize the compressive strength, the R function optim() searches the 
##mixture space for optimal formulation. A custom translation is required to 
##translate a candidate mixture to a prediction. The optim() function returns 
##settings that minimize a function, hence we use negative of compressive 
##strength.

##The below custom function ensures that:
#### (a) the proportions are between 0 and 1.
#### (b) the proportion of water does not fall below 5%.
##If any of these condition are violates the custom function returns a large 
##positive value which the optimization function avoids. The input to the 
##function are a set of 6 mxture proportions and the model used for prediction.

modelPrediction = function(x, mod) {
  ##Check to make sure the proportions are in the correct range:
  if(x[1] < 0 | x[1] > 1) return(10^38)
  if(x[2] < 0 | x[2] > 1) return(10^38)
  if(x[3] < 0 | x[3] > 1) return(10^38)
  if(x[4] < 0 | x[4] > 1) return(10^38)
  if(x[5] < 0 | x[5] > 1) return(10^38)
  if(x[6] < 0 | x[6] > 1) return(10^38)
  
  ##Determine the water proportion and check its value.
  x = c(x, 1 - sum(x))
  if(x[7] < 0.05) return(10^38)
  
  ##Convert x to a dataframe, assign names and set age as 28
  tmp = as.data.frame(x)
  names(tmp) = c("Cement", "BlastFurnaceSlag", "FlyAsh",
               "Superplasticizer", "CoarseAggregate", 
               "FineAggregate", "Water")
  tmp$Age = 28
  ##Get the model prediction and return the negative value.
  -predict(mod, tmp)
}


##First the Cubist model:
cbResults = startingValues
cbResults$Water = NA
cbResuls$Prediction = NA
##Loop over each starting point and conduct the search:
for (i in 1:nrow(cbResults)) {
  
  results = optim(unlist(cbResults[i, 1:6]), modelPrediction,
                  method = "Nelder-Mead", ##Use method = "SANN" for simultaneous
                  ##annealing.
                  control = list(maxit = 5000),
                  ##The next option is passed to the modelPrediction() function.
                  mod = cbFit)
  
  ##Save the predicted compressive strength and the final mixture values.
  cbResults$Prediction[i] = -results$value
  cbResults[i, 1:6] = results$par
}

##Calculate the water proportion.
cbResults$Water = 1 - apply(cbResults[1:6], 1, sum)
##Keep the top 3 mixtures:
cbResults = cbResults[order(-cbResults$Prediction), ][1:3, ]
cbResults$Model = "Cubist"


##Neural Network:
nnetResults <- startingValues
nnetResults$Water <- NA
nnetResults$Prediction <- NA

for(i in 1:nrow(nnetResults))
{
  results <- optim(unlist(nnetResults[i, 1:6,]),
                   modelPrediction,
                   method = "Nelder-Mead",
                   control=list(maxit=5000),
                   mod = nnetFit)
  nnetResults$Prediction[i] <- -results$value
  nnetResults[i,1:6] <- results$par
}
nnetResults$Water <- 1 - apply(nnetResults[,1:6], 1, sum)
nnetResults <- subset(nnetResults, Prediction > 0 & Water > .02)
nnetResults <- nnetResults[order(-nnetResults$Prediction),][1:3,]
nnetResults$Model <- "NNet"

##The Cubist mixtures were predicted to have similar compressive strength. Their
##formulation were differentiated by cement, slag, ash and plasticizer components.
##The neural network mixtures were in a nearby region of mixture space and had 
##predicted values that were lower than the Cubist model but larger than the 
##best-observed mixtures. In each of the six cases, the mixtures have very low 
## proportion of water. 

## Principal component analysis was used to represent the training set mixture
## using two components.
##PCA was conducted on 28-day-old mixtures and the six 
##predicted mixtures were projected. The components are combined and plotted.

pp2 = preProcess(age28Data[, 1:7], "pca") ##Run PCA on 28-day old data.
##Get PCA components:
pca1 = predict(pp2, age28Data[, 1:7])
pca1$Data = "Training Set"
pca1$Data[startPoints] = "Starting Values" ##Label starting points.


################################################################################
### Session Information

sessionInfo()
rm(list = ls())
q("no")

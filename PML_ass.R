## Data 
# The training data for this project are available here: 
#   https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
# The test data are available here:
#   https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv
# The data for this project come from this source: http://groupware.les.inf.puc-rio.br/har. 
# If you use the document you create for this class for any purpose please cite them 
# as they have been very generous in allowing their data to be used for this kind of assignment. 

## What you should submit
# The goal of your project is to predict the manner in which they did the exercise. 
# This is the "classe" variable in the training set. You may use any of the other variables to predict with.
# You should create a report describing how you built your model, how you used cross validation, 
# what you think the expected out of sample error is, and why you made the choices you did. 
# You will also use your prediction model to predict 20 different test cases. 

## Peer Review Portion
# Your submission for the Peer Review portion should consist of a link to a Github repo 
# with your R markdown and compiled HTML file describing your analysis. 
# Please constrain the text of the writeup to < 2000 words and the number of figures to be less than 5. 
# It will make it easier for the graders if you submit a repo with a gh-pages branch so the HTML page 
# can be viewed online (and you always want to make it easy on graders :-).

## Course Project Prediction Quiz Portion
# Apply your machine learning algorithm to the 20 test cases available in the test data above 
# and submit your predictions in appropriate format to the Course Project Prediction Quiz for automated grading. 

## Reproducibility 
# Due to security concerns with the exchange of R code, your code will not be run during the evaluation by your classmates.
# Please be sure that if they download the repo, they will be able to view the compiled HTML version of your analysis. 
# GH-PAGES BRANCH
# Create an index.html file on the local computer
# Commit and push the html file to the remote repository
# Navigate to https://username.github.io/username/reponame to see the website


## Download data
pml_training<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
pml_testing<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")

## EDA
dim(pml_training)
dim(pml_testing)
names(pml_training)
head(pml_training[,1:10])
length(grep("belt", names(pml_training), ignore.case = TRUE))
unique(pml_training$classe)
length(grep("belt|arm|dumb", names(pml_training), ignore.case = TRUE))
names(pml_training[-grep("belt|arm|dumb", names(pml_training), ignore.case = TRUE)])
# [1] "X"                    "user_name"            "raw_timestamp_part_1" "raw_timestamp_part_2" "cvtd_timestamp"      
# [6] "new_window"           "num_window"           "classe"              
# 7 columns describing exercise, 38*4 measurements = 159 cols + classe
dim(pml_training)[2] # 160


unique(pml_training$user_name)
# check for mean or avg in the names: kurtosis, max, min, avg, stddev, 
# accelerometers on the belt, forearm, arm, and dumbell: 38 variables for each
length(grep("forearm", names(pml_training), ignore.case = TRUE))
length(grep("arm", names(pml_training), ignore.case = TRUE))
length(grep("dumb", names(pml_training), ignore.case = TRUE))



str(pml_training[-grep("belt|arm|dumb", names(pml_training), ignore.case = TRUE)])
unique(pml_training$new_window)
# [1] "no"  "yes"
dim(pml_training)[1]/length(unique(pml_training$num_window)) # about 23 rows per num window
length(unique(pml_training$X)) # just an identifier: counts from 1 to num-rows # do not use
range(pml_training$X)
# still need to find out:
# 1. relation between "raw_timestamp_part_1" "raw_timestamp_part_2" "cvtd_timestamp"
#     can I get rid of the raw values?
# 2. what is new_window and num_window
library(dplyr)
pml_training %>%
  group_by(new_window) %>%
  summarise(n = n())
head(pml_training$X[pml_training$new_window == "yes"]) # not regular
pml_training[22:54,6:7] # new window yes row before increasing by 1 num_w
tab_user_window <- pml_training %>%
  group_by(user_name, num_window) %>%
  summarise(n = n())
range(pml_training$num_window) # [1]   1 864
length(unique(pml_training$num_window)) # 858 values
# do users share window numbers?
dim(tab_user_window) # 858 3
length(unique(tab_user_window$num_window)) # 858
# no num_w repeated, hence one num_w goes to only one user

sum(!complete.cases(pml_training)) # 19216 rows is missing vals!!!
na_count <- sapply(pml_training,function(x) sum(is.na(x)))
unique(na_count) # [1]     0 19216
max(na_count)/dim(pml_training)[1] # for many columns 97% of the values are missing
sum(na_count == max(na_count)) # 67 columns with 97% missing vals: we will be left with 93 variables
# 7 are identifiers, so 86 meas vars
na_rows <- names(na_count[na_count == max(na_count)])
# get rid of the word after last "_" after the names
paste0(strsplit(na_rows, "_")[[1]][1], "_", strsplit(na_rows, "_")[[1]][2])
na_meas <- sapply(na_rows, function(x) paste0(strsplit(x, "_")[[1]][1], "_", strsplit(x, "_")[[1]][2]))
unique(na_meas)

# check on which data we have to test: many cols NAs
library(utils)
na_count_tst <- sapply(pml_testing,function(x) sum(is.na(x)))
unique(na_count_tst)
notna_names <- names(na_count_tst[na_count_tst != max(na_count_tst)])
dim(pml_testing)[2]-length(notna_names)
dim(pml_testing[,na_count_tst != max(na_count_tst)])

# near zero variance
nsv <- nearZeroVar(new_training,saveMetrics=TRUE)


new_testing <- pml_testing[, names(pml_testing) %in% c(notna_names,"classe")]
new_training <- pml_training[, names(pml_training) %in% c(notna_names,"classe")]
# actually X and timestamp confuse the prediction. Better removing.
# and new_window is useless
new_training <- new_training[,c(8:60)]
new_testing <- new_testing[,c(8:60)]
# 5 classes
# set them as factors
new_training$classe <- as.factor(new_training$classe)

length(grep("arm", names(new_training)))
length(grep("bell", names(new_training)))
length(grep("belt", names(new_training)))
# 13 variables with measurements
length(complete.cases(new_training)) # no NA's
length(complete.cases(new_testing)) # no NA's
names(new_training)
head(new_training[,1:7])
# box and whisker plots for the classes versus the 4 roll variables 
plot(new_training$classe, new_training$roll_belt, xlab = "Classe", ylab = "Roll Belt") # A mean much lower

plot(new_training$classe, new_training$roll_arm) # same
plot(new_training$classe, new_training$roll_forearm) # C mean higher
plot(new_training$classe, new_training$roll_dumbbell) # C mean lower

##############
## Correlation: remove highly correlated variables (>0.8)

dim(new_training[,-53])
cor_mat = cor(new_training[-53])
hc = findCorrelation(cor_mat, cutoff = 0.8)
hc = sort(hc)
reduced_train = new_training[,-c(hc)]
reduced_test = new_testing[,-c(hc)]
names(reduced_train)

# classification problem
# k = 5 validation

# "rpart" and CV: 51% accuracy in CV
set.seed(11111)
library(caret)
cv_rpart_Fit <- train(classe ~.,data = reduced_Data, method="rpart",
                      trControl = trainControl(method="cv",number=5)) 
cv_rpart_Fit$finalModel
confusionMatrix.train(cv_rpart_Fit) # Accuracy (average) : 0.5121

predictions <- predict(cv_rpart_Fit,newdata = new_testing)
predictions # [1] D A C A A C D A A A C C C A C A A A A C

# "rf" and CV
set.seed(11111)
library(parallel) 
# Calculate the number of cores
no_cores <- detectCores() - 1

library(doParallel)
# create the cluster for caret to use
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)

library(randomForest)
cv_rf_Fit <- train(classe ~.,data = reduced_train, method="rf",
                   trControl = trainControl(method="cv",number=5), allowParallel = TRUE) 
stopCluster(cl)
registerDoSEQ()

cv_rf_Fit$finalModel # OOB estimate of  error rate: 0.51%
confusionMatrix.train(cv_rf_Fit) # Accuracy (average) : 0.9928

predictions <- predict(cv_rf_Fit,newdata = reduced_test)
predictions # [1] B A B A A E D B A A B C B A E E A B B B
pml_testing$num_window # [1]  74 431 439 194 235 504 485 440 323 664 859 461 257 408 779 302  48 361  72 255



AAA = table(pml_training$num_window, pml_training$classe)
AAA[rownames(AAA) %in% pml_testing$num_window,]
pml_testing$num_window
rownames(AAA)

class(AAA)#[new_training$num_window %in% new_testing$num_window]
new_training$classe[new_training$num_window == 74]
unique(AAA)

getTree(cv_rf_Fit$finalModel,k=2)


dim(new_training)
m <- abs(cor(new_training[,-c(1,54)]))







###############################################
## Download data
pml_training<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
pml_testing<- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
# EDA and tidying
dimTr <- dim(pml_training)
dimTe <- dim(pml_testing)
unique(pml_training$classe) # 5 classes
meas <- length(grep("belt|arm|dumb", names(pml_training), ignore.case = TRUE)) # 38 measurements columns per sensor
# Check NAs in train
na_count <- sapply(pml_training,function(x) sum(is.na(x)))
unique(na_count) # [1]     0 19216
prop_missing <- max(na_count)/dim(pml_training)[1] # for many columns 97% of the values are missing
cols_missing <- sum(na_count == max(na_count)) # 67 columns with 97% missing vals: we will be left with 93 variables
# Check NAs in test
na_count_tst <- sapply(pml_testing,function(x) sum(is.na(x)))
unique(na_count_tst) # 0 20
notna_names <- names(na_count_tst[na_count_tst != max(na_count_tst)]) # cols with values
not_NA_cols <- length(notna_names) # cols with values
# Remove columns: first the empty test columns, then the identifier columns
new_testing <- pml_testing[, names(pml_testing) %in% c(notna_names,"classe")]
new_training <- pml_training[, names(pml_training) %in% c(notna_names,"classe")]
new_training <- new_training[,c(8:60)]
new_testing <- new_testing[,c(8:60)]
num_cols <- dim(new_training)[2]

cor_mat = cor(new_training[-53])
hc = findCorrelation(cor_mat, cutoff = 0.8)
hc = sort(hc)
reduced_train = new_training[,-c(hc)]
reduced_test = new_testing[,-c(hc)]
red_cols <- dim(reduced_train)[2]
reduced_train$classe <- as.factor(reduced_train$classe)

boxplot(reduced_train$roll_forearm ~ reduced_train$classe, xlab = "classe", ylab = "roll_forearm") # C median higher



# gbm booster predictor
library(gbm)

start_gbm <- Sys.time()
set.seed(11111)
fit_B <- train(classe ~., method = "gbm", data = reduced_train, verbose = FALSE, trControl = trainControl(method="cv",number=5))
end_gbm <- Sys.time()
end_gbm - start_gbm # Time difference of 4.775969 mins

CM_B <- confusionMatrix.train(fit_B)
sum(diag(CM_B$table)) # Accuracy (average) : 95.35

predictions <- predict(fit_B,newdata = reduced_test)
predictions # C A B A A E D B A A B C B A E E A B A B

# random forest
library(parallel) 
# Calculate the number of cores for parallel computation
no_cores <- detectCores() - 1
library(doParallel)
# create the cluster for caret to use
cl <- makePSOCKcluster(no_cores)
registerDoParallel(cl)
library(randomForest)
set.seed(11111)

start_rf <- Sys.time()
cv_rf_Fit <- train(classe ~.,data = reduced_train, method="rf",
                   trControl = trainControl(method="cv",number=5), allowParallel = TRUE) 
end_rf <- Sys.time()
end_rf - start_rf # Time difference of 8.467381 mins

stopCluster(cl)
registerDoSEQ()
CM_rf = confusionMatrix.train(cv_rf_Fit)
sum(diag(CM_rf$table)) # Accuracy (average) : 0.9932

cv_rf_Fit$finalModel # OOB estimate of  error rate: 0.51%

CM_count_rf <- cv_rf_Fit$finalModel$confusion
round((sum(CM_count_rf)-sum(diag(CM_count_rf)))/sum(CM_count_rf)*100,2)

predictions <- predict(cv_rf_Fit,newdata = reduced_test)
predictions # B A B A A E D B A A B C B A E E A B B B



# "rpart" and CV: 51% accuracy in CV
set.seed(11111)
library(caret)
cv_rpart_Fit <- train(classe ~.,data = reduced_train, method="rpart",
                      trControl = trainControl(method="cv",number=5)) 
str(cv_rpart_Fit)
cv_rpart_Fit$
CM_rpart <- confusionMatrix.train(cv_rpart_Fit)
sum(diag(CM_rpart$table)) # Accuracy (average) : 50.38

predictions <- predict(cv_rpart_Fit,newdata = reduced_test)
predictions # D A C A A C D A A A C C C A C A A A A C

# glm?

# lda
fit_LDA <- train(classe ~., method = "lda", data = reduced_train, trControl = trainControl(method="cv",number=5))
CM_LDA <- confusionMatrix.train(fit_LDA)
sum(diag(CM_LDA$table)) # Accuracy (average) : 64.25

predictions <- predict(fit_LDA,newdata = reduced_test)
predictions # B A A C C C D D A A D A B A E A A B B B




as.data.frame(cv_rpart_Fit$finalModel$confusion)
CM_rpart <- as.table(confusionMatrix(reduced_train$classe, predict(cv_rpart_Fit, newdata = reduced_train)))
CM_rpart$table
ncol(CM_rpart)+1

CM_count_rpart <- confusionMatrix(reduced_train$classe, predict(cv_rpart_Fit, newdata = reduced_train))
CM_count_LDA <- confusionMatrix(reduced_train$classe, predict(fit_LDA, newdata = reduced_train))

library(plyr)
library(dplyr)
library(caret)
library(gbm)
library(randomForest)

setwd("C:/Users/cberi/Desktop/Development_Time/Coursera Courses/Johns Hopkins Data Science/Practical Machine Learning/Project")

train <- read.csv("pml-training.csv",stringsAsFactors = TRUE)
test <- read.csv("pml-testing.csv",stringsAsFactors = TRUE)

glimpse(train)
glimpse(test)


table(train$classe)
#There are 5 possible values for classe, which is waht we are trying to predict
#This is a categorical variable, so we should keep that in mind for our choice of algorithm


ncol(train)
#There are 160 columns to deal with and consider using for modeling purposes




############################################ INITIAL EXPLORATION 1: Cramer's V ############################

#Cramer's V to assess correlation between classe and other variables in the dataset
#http://sas-and-r.blogspot.com/2011/06/example-839-calculating-cramers-v.html
cv.test = function(x,y) {
      CV = sqrt(chisq.test(x, y, correct=FALSE)$statistic /
                      (length(x) * (min(length(unique(x)),length(unique(y))) - 1)))

      return(as.numeric(CV))
}


#https://en.wikipedia.org/wiki/Cramer's_V
#Should be between 0 and 1?

var <- c()
cramer <- c()

for(i in 1:ncol(train)){

      var[i] = names(train)[i]
      cramer[i] = cv.test(train$classe,train[,i])
      
}

cramers_v <- data.frame(var,cramer) %>% 
      arrange(desc(cramer))

glimpse(cramers_v)

write.csv(cramers_v,"Cramers_V.csv",row.names=FALSE)





################################ TRYING TO MODEL ##############################



#Remove all variables with blank or missing values for modeling purposes
modeling_data = train
flag_for_removal <- c()
i = 1
j= 1
for(i in 1:ncol(train)){
      
      
      if(sum(train[,i]=="") > 0 | sum(is.na(train[,i])==TRUE) >0){
            flag_for_removal[j] = i
            j = j +1
      }
      
}



modeling_data <- train[,-flag_for_removal]
#Also need to remove variables which seem to be beyond reasonably highly correlated with classe
#Also removing user name because from implementation perspective, we want to detect from
#other metrics
modeling_data <- modeling_data %>% select(-X,-num_window,-raw_timestamp_part_1,-raw_timestamp_part_2,-cvtd_timestamp,
                                          -user_name,-new_window)


modeling_data_pre_cut <- modeling_data

#To save on computational time and avoid overfitting, group all numeric variables into quintiles
for(i in 1:(ncol(modeling_data)-1)){
      
    modeling_data[,i] =  cut(modeling_data[,i], breaks=c(quantile(modeling_data[,i], probs = seq(0, 1, by = 0.20))), 
          include.lowest=TRUE)

}


glimpse(modeling_data)
#End of creation of modeling dataset



predictors <- modeling_data %>% select(-classe)
classe <- modeling_data$classe

fit_1 <- train(predictors,classe,method="rf",ntree=50)






#################################### PREDICT ON TEST DATASET #############################


flag_for_removal <- c()
i = 1
j= 1
for(i in 1:ncol(test)){
      
      
      if(sum(test[,i]=="") > 0 | sum(is.na(test[,i])==TRUE) >0){
            flag_for_removal[j] = i
            j = j +1
      }
      
}



test <- test[,-flag_for_removal]
#Also need to remove variables which seem to be beyond reasonably highly correlated with classe
#Also removing user name because from implementation perspective, we want to detect from
#other metrics
test <- test %>% select(-X,-num_window,-raw_timestamp_part_1,-raw_timestamp_part_2,-cvtd_timestamp,
                                          -user_name,-new_window)

#To save on computational time and avoid overfitting, group all numeric variables into quintiles
for(i in 1:(ncol(test)-1)){
      
     test[,i] = cut(test[,i], breaks=c(quantile(modeling_data_pre_cut[,i], probs = seq(0, 1, by = 0.20))), 
          include.lowest=TRUE)
      
}



predictions_fit_1 <- predict(fit_1,test,ntree=50)
#Ended up getting 95% of classe's right on out of sample dataset
#Given this out-of-sample accuracy, I did not go back and re-visit models that use cross-validation
#for parameter tuning!
#Predictions based on algorithm with no cross-validation
#1) C
#2) A
#3) B
#4) A
#5) A
#6) E
#7) D
#8) B
#9) A
#10) A
#11) B
#12) C
#13) B
#14) A
#15) E
#16) E
#17) A
#18) B
#19) B
#20) B

predictions_fit_1






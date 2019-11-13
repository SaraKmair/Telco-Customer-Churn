---
  title: "Customer_churn"
author: "Sara Kmair"
date: "10/22/2019"
output: html_document
---

library(plyr)
library(dplyr)
library(ggplot2)
library(ggthemes)
library(partitions)
library(caret)
library(randomForest)
library(ROSE)
library(naivebayes)
library(psych)




cchurn <- read.csv("Customer-Churn.csv")
str(cchurn)
summary(cchurn)

```

#we notice that total charges the only attribute that has missing values (11 NAs)

#Dealing with the missing values 

#when we take a look at the missing value in total charges attribute we notice that all the tenure for these customer is 0 which might mean that these customers are new customers and they haven't been charges yet as they might have been with the company for less than a month 

#I checked here if a customer has been with the company for a month. I noticed that monthly charge = total charge
cchurn[which(cchurn$tenure == 1), ]

#I will replace the total charges missing values with 0 for now 
cchurn[which(is.na(cchurn$TotalCharges)), ] 

#replacing the missing values with 0
cchurn$TotalCharges[which(is.na(cchurn$TotalCharges))] <- cchurn$MonthlyCharges[which(is.na(cchurn$TotalCharges))] 
```





#Changing Customer ID to a character 

cchurn$customerID <- as.character(cchurn$customerID)
str(cchurn)
#changing senior citizen into a factor 
cchurn$SeniorCitizen <- as.factor(cchurn$SeniorCitizen)
summary(cchurn$SeniorCitizen)

#The data set has 21 variables. 18 of them are categorical 


#dropping customer ID column since it has no influence 
cchurn <- subset(cchurn, select = -c(customerID))

names(cchurn)
descriptive <- c("gender", "SeniorCitizen", "Partner", "Dependents" ,"PhoneService", "MultipleLines", "InternetService", "OnlineSecurity", "DeviceProtection","TechSupport", "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod")

#This step is univariate analysis 


#Let's see who are our customer 

for (i in descriptive)
{
  print(barplot(table(cchurn[, i]), xlab = colnames(cchurn[i]), col = c("pink", "purple", "cyan", "gold")))
}

#Bivariate Analysis 

#the relationship between each attribute and the churn 
#The count of gender and frequency and the relationship with customer churn or not 
cchurn %>%
  group_by(Churn, gender) %>%
  summarize(count = n())%>%
  mutate(Frequency = count / sum(count))%>%
  ggplot(., aes(x = gender, y = Frequency, fill = Churn)) +
  geom_col( position ="dodge" )





#the relationship between each attribute and the churn 
#The count of SeniorCirizen and frequency and the relationship with customer churn or not 

cchurn %>%
  group_by(Churn, SeniorCitizen) %>%
  summarize(count = n())%>%
  mutate(Frequency = count / sum(count))%>%
  ggplot(., aes(x = SeniorCitizen, y = Frequency, fill = Churn)) +
  geom_col( position ="dodge" )





#the relationship between each attribute and the churn 
#The count of techsupport and frequency and the relationship with customer churn or not 
cchurn %>%
  group_by(Churn, TechSupport) %>%
  summarize(count = n())%>%
  mutate(Frequency = count / sum(count)) %>%
  ggplot(., aes(x = TechSupport, y = Frequency, fill = Churn)) +
  geom_col(position = "dodge")


#the relationship between each attribute and the churn 
#The count of techsupport and frequency and the relationship with customer churn or not 
cchurn %>%
  group_by(Churn, Partner) %>%
  summarize(count = n())%>%
  mutate(Frequency = count / sum(count)) %>%
  ggplot(., aes(x = Partner, y = Frequency, fill = Churn)) +
  geom_col(position = "dodge")




#the relationship between each attribute and the churn 
#The count of techsupport and frequency and the relationship with customer churn or not 
cchurn %>%
  group_by(Churn, Dependents) %>%
  summarize(count = n())%>%
  mutate(Frequency = count / sum(count)) %>%
  ggplot(., aes(x = Dependents, y = Frequency, fill = Churn)) +
  geom_col(position = "dodge")



#the relationship between each attribute and the churn 
#The count of techsupport and frequency and the relationship with customer churn or not 
cchurn %>%
  group_by(Churn, MultipleLines) %>%
  summarize(count = n())%>%
  mutate(Frequency = count / sum(count)) %>%
  ggplot(., aes(x = MultipleLines, y = Frequency, fill = Churn)) +
  geom_col(position = "dodge")

#let's subset the customer churn and see if they have anything in common 
churn_sub <- cchurn[which(cchurn$Churn == 'Yes'), ] 
#subsetting the dataset 
nochurn_sub <- cchurn[which(cchurn$Churn == 'No'), ] 

#Higher monthly charges might be one of the reasons of customer churn 

boxplot(churn_sub$MonthlyCharges, nochurn_sub$MonthlyCharges, names = c("Churn", "Not Churn"), col = c("aquamarine4", "aquamarine1"), horizontal = TRUE, xlab = "Monthly Charges")


#Notice that the churn monthly charge is always higher in all kind of contracts too 
ggplot(cchurn, aes(x = Contract, y = MonthlyCharges, fill = Churn)) +
  geom_boxplot( ) +
  coord_flip() +
  theme_economist()

#We can coclude that the higher the monthly charge the more likely for the customer to churn 


#Percentage of customer churn
churn_count <- table(cchurn$Churn)
churn_percent <- table(cchurn$Churn) / length(cchurn$Churn)
barplot(churn_count, main = "Customer Churn", col = "cyan3")
#notice the the data is imbalance 




We notice that the data imbalanced.

imbalanced data refers to classification problems where we have unequal instances for different classes. Most machine learning classification algorithms are sensitive to unbalance in the predictor classes. 

Under-sampling, we randomly select a subset of samples from the class with more instances to match the number of samples coming from each class. 
oversampling, we randomly duplicate samples from the class with fewer instances or we generate additional instances based on the data that we have, so as to match the number of samples in each class. 

```{r}
#checking the severity of imbalanced data 
table(cchurn$Churn)

#Dividing our dataset into training and test set then deal with the imbalanced data 
set.seed(100)
index <- sample(2, nrow(cchurn), replace = TRUE, prob = c(0.7, 0.3)) #in this case 70% of the data will go for training 
train_data <- cchurn[index==1, ] #subsetting the training data 70% 
test_data  <- cchurn[index==2,  ] #subsetting the test data 30%





#imbalanced data: means the predicting target class distribution is skewed. balancing the data will avoid over or under sampling 
#balancing the data 
nrow(train_data)
balanced <- ovun.sample(Churn~., data = train_data, method = "both", p = 0.5)$data
table(both$Churn)




# Developing predictive model 
# Predictive model: Random Forest
rf_model <- randomForest(Churn~., data = balanced)

#evaluating the model
confusionMatrix(predict(rf_model, test_data), test_data$Churn, positive = "Yes")

#Model accuracy 0.77



#Developing predictive model 
#Predictive model: Naive Bayes
nb_model <- naive_bayes(Churn~., data = balanced)

#evaluating the model 
confusionMatrix(predict(nb_model,test_data), test_data$Churn, positive = "Yes")

#model accuracy 0.7



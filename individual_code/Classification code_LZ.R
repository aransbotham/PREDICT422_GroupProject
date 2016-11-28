########################################################################
#Name: Group 2 - Bruckner, Funk, Sheets, Zimmerman                     #
#Due Date: December 4th, 2016                                          #
#Course: PREDICT 422                                                   #
#Section: 56                                                           #
#Assignment: Group Project                                             #
#Goal: We would like to develop a classification model using data from #
#      the most recent campaign that can effectively captures likely   #
#      donors so that the expected net profit is maximized. We would   #
#      also like to build a prediction model to predict expected gift  #
#      amounts from donors – the data for this will consist of the     #
#      records for donors only.                                        #
########################################################################

#Variables
# ID number [Do NOT use this as a predictor variable in any models]
# REG1, REG2, REG3, REG4: Region (There are five geographic regions; A 1 indicates the potential donor belongs to this region.)
# HOME: (1 = homeowner, 0 = not a homeowner)
# CHLD: Number of children
# HINC: Household income (7 categories)
# GENF: Gender (0 = Male, 1 = Female)
# WRAT: Wealth Rating (Wealth rating uses median family income and population statistics from each area to index relative wealth within each state. The segments are denoted 0-9, with 9 being the highest wealth group and 0 being the lowest.)
# AVHV: Average Home Value in potential donor's neighborhood in $ thousands
# INCM: Median Family Income in potential donor's neighborhood in $ thousands
# INCA: Average Family Income in potential donor's neighborhood in $ thousands
# PLOW: Percent categorized as “low income” in potential donor's neighborhood
# NPRO: Lifetime number of promotions received to date
# TGIF: Dollar amount of lifetime gifts to date
# LGIF: Dollar amount of largest gift to date
# RGIF: Dollar amount of most recent gift
# TDON: Number of months since last donation
# TLAG: Number of months between first and second gift
# AGIF: Average dollar amount of gifts to date
# DONR: Classification Response Variable (1 = Donor, 0 = Non-donor)
# DAMT: Prediction Response Variable (Donation Amount in $).

setwd("/Users/asheets/Documents/Work_Computer/Grad_School/PREDICT_422/PREDICT422_GroupProject")


set.seed(1)

#Install Packages if they don't current exist on this machine
list.of.packages <- c("doBy"
                      ,"psych"
                      ,"lars"
                      ,"GGally"
                      ,"ggplot2"
                      ,"gridExtra"
                      ,"corrgram"
                      ,"corrplot"
                      ,"leaps"
                      ,"glmnet"
                      ,"MASS"
                      ,"gbm"
                      ,"tree"
                      ,"rpart"
                      ,"rpart.plot"
                      ,"gam"
                      ,"class"
                      ,"e1071"
                      ,"ggplot2")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

library(doBy)
library(psych)
library(lars)
library(GGally)
library(ggplot2)
library (gridExtra)
library(corrgram)
library(corrplot)
library(leaps)
library(glmnet)
library(MASS)
library(gbm)
library(tree)
library(rpart)
library(rpart.plot)
library(gam)
library(class)
library(e1071)
library(randomForest)

# Load the diabetes data
data <- read.csv("~/Documents/Northwestern/PRED 422/Group Project/charity.csv")

#Explore the data -- how big is it, what types of variables included, distributions and missing values.
dim(data)
summary(data) # donr and damt each have 2007 NA values -- these are the test set
str(data) # all int except agif is num and part is a factor with 3 levels
head(data)
class(data) # data.frame
nrow(data) # 8009 rows
ncol(data) # 24 variables
names(data)

plots <- vector("list", 22)
names <- colnames(data)

#Visualize the variables
plot_vars = function (data, column)
  ggplot(data = data, aes_string(x = column)) +
  geom_histogram(color =I('black'),fill = I('#099009'))+
  xlab(column)

plots <- lapply(colnames(data)[2:23], plot_vars, data = data[2:23])

n <- length(plots)
nCol <- floor(sqrt(n))
do.call("grid.arrange", c(plots))

#There is no missing data
#Some variables are heavily skewed in one direction
#Some variable are bimodal

##Transform some variables
charity.t <- data

#These variables are all skewed right
charity.t$avhv <- log(charity.t$avhv)
charity.t$inca <- log(charity.t$inca)
charity.t$incm <- log(charity.t$incm)
charity.t$agif <- log(charity.t$agif)
charity.t$rgif <- log(charity.t$rgif)
charity.t$lgif <- log(charity.t$lgif)
charity.t$tgif <- log(charity.t$tgif)

#Visualize data after making transformations
plots2 <- lapply(colnames(charity.t)[2:23], plot_vars, data = charity.t[2:23])

n <- length(plots2)
nCol <- floor(sqrt(n))
do.call("grid.arrange", c(plots2))

#Examine Correlations
M <- cor(charity.t[sapply(charity.t, is.numeric)],use="complete.obs")
#
correlations <- data.frame(cor(charity.t[sapply(charity.t, is.numeric)],use="complete.obs"))
#
significant.correlations <- data.frame(
  var1 = character(),
  var2 = character(),
  corr = numeric())
#
for (i in 1:nrow(correlations)){
  for (j in 1:ncol(correlations)){
    tmp <- data.frame(
      var1 = as.character(colnames(correlations)[j]),
      var2 = as.character(rownames(correlations)[i]),
      corr = correlations[i,j])
    #
    if (!is.na(correlations[i,j])) {
      if (correlations[i,j] > .5 & as.character(tmp$var1) != as.character(tmp$var2)
          | correlations[i,j] < -.5 & as.character(tmp$var1) != as.character(tmp$var2) ) {
        significant.correlations <- rbind(significant.correlations,tmp) }
    }
  }
}

significant.correlations <- significant.correlations[order(abs(significant.correlations$corr),decreasing=TRUE),] 
significant.correlations <- significant.correlations[which(!duplicated(significant.correlations$corr)),]
significant.correlations

##Results with tgif transformed:
#var1 var2       corr
#24 damt donr  0.9817018
#15 tgif npro  0.8734276
#17 rgif lgif  0.8512241
#4  inca avhv  0.8484572
#7  inca incm  0.8296747
#18 agif lgif  0.8294224
#8  plow incm -0.8120381
#20 agif rgif  0.7706645
#11 plow inca -0.7510141
#3  incm avhv  0.7304313
#5  plow avhv -0.7187952
#2  damt chld -0.5531045
#1  donr chld -0.5326077


#Run t-test to test group means for all numeric variables across classification outcome
significant2 <- data.frame(var = character(),
                           p_value = numeric())

numeric_vars <- data.frame(var = colnames(data)[-22:-24])

for (i in 2:dim(numeric_vars)[1]) {
  test <- t.test(data[which(data$donr == 1),c(numeric_vars$var[i])],data[which(data$donr== 0),c(numeric_vars$var[i])], "g", 0, FALSE, TRUE, 0.95)
  if (!is.na(test$p.value) & test$p.value <= 0.05) {
    tmp <- data.frame(var=numeric_vars$var[i],
                      p_value = round(test$p.value,4))
    
    significant2 <- rbind(significant2,tmp)
  }
  print(i)
}

significant2
##There are several variables whose group means are significantly different between donr = 1 and donr = 0
# var p_value
# 1  reg1  0.0000
# 2  reg3  0.0000
# 3  reg4  0.0000
# 4  home  0.0000
# 5  chld  0.0000
# 6  avhv  0.0004
# 7  inca  0.0037
# 8  plow  0.0000
# 9  npro  0.0000
# 10 lgif  0.0000
# 11 agif  0.0192

##some multi-collinearity present between variables, should be kept in mind for further analysis

##Visualize this
corrgram(charity.t, order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="Correlations")

corrplot(M, method = "square") #plot matrix

# set up data for analysis

data.train <- charity.t[charity.t$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity.t$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity.t$part=="test",]
n.test <- dim(data.test)[1] # 2007
x.test <- data.test[,2:21]

x.train.mean <- apply(x.train, 2, mean)
x.train.sd <- apply(x.train, 2, sd)
x.train.std <- t((t(x.train)-x.train.mean)/x.train.sd) # standardize to have zero mean and unit sd
apply(x.train.std, 2, mean) # check zero mean
apply(x.train.std, 2, sd) # check unit sd
data.train.std.c <- data.frame(x.train.std, donr=c.train) # to classify donr
data.train.std.y <- data.frame(x.train.std[c.train==1,], damt=y.train) # to predict damt when donr=1

x.valid.std <- t((t(x.valid)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.valid.std.c <- data.frame(x.valid.std, donr=c.valid) # to classify donr
data.valid.std.y <- data.frame(x.valid.std[c.valid==1,], damt=y.valid) # to predict damt when donr=1

x.test.std <- t((t(x.test)-x.train.mean)/x.train.sd) # standardize using training mean and sd
data.test.std <- data.frame(x.test.std)

#Visualize standardized data

plots3 <- lapply(colnames(data.train.std.c)[1:21], plot_vars, data = data.train.std.c[1:21])

n <- length(plots3)
nCol <- floor(sqrt(n))
do.call("grid.arrange", c(plots3))

#########################################################################################
#Question 2 -- CLASSIFICATION MODELS                                                    #
#########################################################################################

#########################################
#    MODEL 1: Logistic                  #
#########################################
set.seed(1)
model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))


post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid1 <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid1, max(profit.log1)) # report number of mailings and maximum profit
#[1]  1330 11637

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid1+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table
#                 c.valid
# chat.valid.log1   0   1
# 0 675  13
# 1 344 986
# check n.mail.valid = 355+988 = 1343
# check profit = 14.5*988-2*1343 = 11640

#####################
#### Model 1a    ####
#####################
#use only those variables found to be significant
set.seed(1)
model.log1a <- glm(donr ~ reg1 + reg3 + reg4 + home + chld + avhv + inca + plow + npro + lgif + agif,
                  data.train.std.c, family=binomial("logit"))

post.valid.log1a <- predict(model.log1a, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1a <- cumsum(14.5*c.valid[order(post.valid.log1a, decreasing=T)]-2)
plot(profit.log1a) # see how profits change as more mailings are made
n.mail.valid1a <- which.max(profit.log1a) # number of mailings that maximizes profits
c(n.mail.valid1a, max(profit.log1a)) # report number of mailings and maximum profit
#[1]  1583 11073

cutoff.log1a <- sort(post.valid.log1a, decreasing=T)[n.mail.valid1a+1] # set cutoff based on n.mail.valid
chat.valid.log1a <- ifelse(post.valid.log1a>cutoff.log1a, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1a, c.valid) # classification table
#                 c.valid
#chat.valid.log1a   0   1
#               0 418  17
#               1 601 982
#601+982=1583
#14.5*982-2*1583=11073

#Resulting model is not more profitable than Model 1. 

#####################
#### Model 1b    ####
#####################
#Try logistic model with additional terms

set.seed(1)
model.log1b <- glm(donr~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2)
                     + avhv + incm + I(incm^2) + inca + I(inca^2) + plow + npro + tgif + I(tgif^2) + lgif +I(lgif^2)
                     + rgif + I(rgif^2) + tdon + tlag + agif + I(agif^2),data.train.std.c, family=binomial("logit"))

post.valid.log1b <- predict(model.log1b, data.valid.std.c, type="response") # n.valid post probs

profit.log1b <- cumsum(14.5*c.valid[order(post.valid.log1b, decreasing=T)]-2)
plot(profit.log1b) # see how profits change as more mailings are made
n.mail.valid1b <- which.max(profit.log1b) # number of mailings that maximizes profits
c(n.mail.valid1b, max(profit.log1b)) # report number of mailings and maximum profit
#[1]  1583 11073

cutoff.log1b <- sort(post.valid.log1b, decreasing=T)[n.mail.valid1b+1] # set cutoff based on n.mail.valid
chat.valid.log1b <- ifelse(post.valid.log1b>cutoff.log1b, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1b, c.valid) # classification table

#Use backward subset selection on model.log1b
regfit.model1b.bwd<-regsubsets(donr~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2)
                               + avhv + incm + I(incm^2) + inca + I(inca^2) + plow + npro + tgif + I(tgif^2) + lgif +I(lgif^2)
                               + rgif + I(rgif^2) + tdon + tlag + agif + I(agif^2),data=data.train.std.c,nvmax=30,
                               method="backward")
summary(regfit.model1b.bwd)

#Create another logistic model using only the top 20 variables from the backward subset selection results
model.log1b_r1<-glm(donr~ reg1 + reg2 + home + chld + I(hinc^2) + genf + wrat + I(wrat^2) + incm + inca + 
                      I(inca^2) + plow + tgif + I(tgif^2) + lgif + I(rgif^2) + tdon + tlag + agif,
                    data.train.std.c, family=binomial("logit"))

post.valid.log1b_r1 <- predict(model.log1b_r1, data.valid.std.c, type="response") # n.valid post probs

profit.log1b_r1 <- cumsum(14.5*c.valid[order(post.valid.log1b_r1, decreasing=T)]-2)
plot(profit.log1b_r1) # see how profits change as more mailings are made
n.mail.valid1b_r1 <- which.max(profit.log1b_r1) # number of mailings that maximizes profits
c(n.mail.valid1b_r1, max(profit.log1b_r1)) # report number of mailings and maximum profit
#[1]  1304.0 11645.5

#20 variable logistic model (model.log1b_r1) is most profitable 


#########################################
#    MODEL 2: Logistic GAM              #
#########################################

model.gam1=gam(donr~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                 avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + s(agif ,df =5),
               family = binomial , data = data.train.std.c)

post.valid.gam1 <- predict(model.gam1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.gam1 <- cumsum(14.5*c.valid[order(post.valid.gam1, decreasing=T)]-2)
plot(profit.gam1) # see how profits change as more mailings are made
n.mail.valid1 <- which.max(profit.gam1) # number of mailings that maximizes profits
c(n.mail.valid1, max(profit.gam1)) # report number of mailings and maximum profit
# 1446.0 11419.5

cutoff.gam1 <- sort(post.valid.gam1, decreasing=T)[n.mail.valid1+1] # set cutoff based on n.mail.valid
chat.valid.gam1 <- ifelse(post.valid.gam1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gam1, c.valid) # classification table
# c.valid
# chat.valid.gam1   0   1
#                0 560  12
#                1 459 987
# check n.mail.valid = 459+987 = 1446
# check profit = 14.5*987-2*1446 = 11419.5


######################
###   Model 2a    ###
######################
#Run logistic GAM model with 20 variables from best subset selection in previous section
model.gam2a=gam(donr~ reg1 + reg2 + home + chld + I(hinc^2) + genf + wrat + I(wrat^2) + incm + inca + 
                  I(inca^2) + plow + tgif + I(tgif^2) + lgif + I(rgif^2) + tdon + tlag + agif, family=binomial,
                data=data.train.std.c)
post.valid.gam2a <- predict(model.gam2a, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.gam2a <- cumsum(14.5*c.valid[order(post.valid.gam2a, decreasing=T)]-2)
plot(profit.gam2a) # see how profits change as more mailings are made
n.mail.valid2a <- which.max(profit.gam2a) # number of mailings that maximizes profits
c(n.mail.valid2a, max(profit.gam2a)) # report number of mailings and maximum profit
# 1302.0 11649.5
cutoff.gam2a <- sort(post.valid.gam2a, decreasing=T)[n.mail.valid2a+1] # set cutoff based on n.mail.valid
chat.valid.gam2a <- ifelse(post.valid.gam2a>cutoff.gam2a, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gam2a, c.valid)
#                 c.valid
#chat.valid.gam2a   0   1
#               0 700  16
#               1 319 983
#319+983=1302
#14.5*983-2*1302=11649.5

#Model model.gam2a more profitable


#########################################
#    MODEL 3: LDA                       #
#########################################

model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) # include additional terms on the fly using I()

# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
# 1391.0 11631

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table
#               c.valid
#chat.valid.lda1   0   1
#              0 622  5
#              1 397 994
# check n.mail.valid = 397 + 994 = 1391
# check profit = 14.5*994-2*1391 = 11631

######################
### Model 3a      ####
######################
#Run another LDA model with 20 best subset selection variables from first section
model.lda3a <- lda(donr~ reg1 + reg2 + home + chld + I(hinc^2) + genf + wrat + I(wrat^2) + incm + inca + 
                    I(inca^2) + plow + tgif + I(tgif^2) + lgif + I(rgif^2) + tdon + tlag + agif, 
                  data.train.std.c) # include additional terms on the fly using I()

# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

post.valid.lda3a <- predict(model.lda3a, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda3a <- cumsum(14.5*c.valid[order(post.valid.lda3a, decreasing=T)]-2)
plot(profit.lda3a) # see how profits change as more mailings are made
n.mail.valid3a <- which.max(profit.lda3a) # number of mailings that maximizes profits
c(n.mail.valid3a, max(profit.lda3a)) # report number of mailings and maximum profit
# 336.0 11639.5

cutoff.lda3a <- sort(post.valid.lda3a, decreasing=T)[n.mail.valid3a+1] # set cutoff based on n.mail.valid
chat.valid.lda3a <- ifelse(post.valid.lda3a>cutoff.lda3a, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda3a, c.valid)
#                 c.valid
#chat.valid.lda3a   0   1
#               0 670  12
#               1 349 987
#349+987=1336
#14.5*987-2*1336=11639.5

#Model 3a marginally more profitable than Model 3


#########################################
#    MODEL 4: QDA                       #
#########################################

model.qda1 =qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,data= data.train.std.c)

# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

post.valid.qda1 <- predict(model.qda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.qda1 <- cumsum(14.5*c.valid[order(post.valid.qda1, decreasing=T)]-2)
plot(profit.qda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda1)) # report number of mailings and maximum profit
# 1369.0 11196.5

cutoff.qda1 <- sort(post.valid.qda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda1 <- ifelse(post.valid.qda1>cutoff.qda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda1, c.valid) # classification table
#                  c.valid
# chat.valid.qda1   0   1
#               0 611  38
#               1 408 961
# check n.mail.valid = 408+961 = 1369
# check profit = 14.5*971-2*1369 = 11341.5

####################
##  Model4a       ##
####################
model.qda4a =qda(donr~ reg1 + reg2 + home + chld + I(hinc^2) + genf + wrat + I(wrat^2) + incm + inca + 
                   I(inca^2) + plow + tgif + I(tgif^2) + lgif + I(rgif^2) + tdon + tlag + agif,
                 data= data.train.std.c)

# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

post.valid.qda4a <- predict(model.qda4a, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.qda4a <- cumsum(14.5*c.valid[order(post.valid.qda4a, decreasing=T)]-2)
plot(profit.qda4a) # see how profits change as more mailings are made
n.mail.valid4a <- which.max(profit.qda4a) # number of mailings that maximizes profits
c(n.mail.valid4a, max(profit.qda4a)) # report number of mailings and maximum profit
# 1421 11107

cutoff.qda4a <- sort(post.valid.qda4a, decreasing=T)[n.mail.valid4a+1] # set cutoff based on n.mail.valid
chat.valid.qda4a <- ifelse(post.valid.qda4a>cutoff.qda4a, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda4a, c.valid) # classification table
#               c.valid
#chat.valid.qda4a   0   1
#               0 560  37
#               1 459 962
#459+962=1421
#14.5*964-2*1421=11136

#Model 4a performed significantly worse than Model 4.

#########################################
#    MODEL 5: KNN                       #
#########################################

set.seed(1)
model.knn1=knn(x.train.std,x.valid.std,c.train,k=1)
mean(c.valid != model.knn1)
# [1] 0.2215064

table(model.knn1 ,c.valid)
#             c.valid
# model.knn1   0   1
#          0 732  160
#          1 287 839

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.knn1 <- cumsum(14.5*c.valid[order(model.knn1, decreasing=T)]-2)
plot(profit.knn1) # see how profits change as more mailings are made

mean((as.numeric(as.character(model.knn1)) - c.valid)^2)
# [1] 0.2215064

# check n.mail.valid = 287+839 = 1126
# check profit = 14.5*839-2*1126 = 9913.5

#####################
#   Model 5a        #
#####################
#Set K=10 for KNN model 5a
set.seed(1)
model.knn5a=knn(x.train.std,x.valid.std,c.train,k=10)
mean(c.valid != model.knn5a)
# [1] 0.1828543

table(model.knn5a ,c.valid)
#           c.valid
#model.knn5a   0   1
#           0 709  59
#           1 310 940


profit.knn5a <- cumsum(14.5*c.valid[order(model.knn5a, decreasing=T)]-2)
plot(profit.knn5a) # see how profits change as more mailings are made

mean((as.numeric(as.character(model.knn5a)) - c.valid)^2)
#0.1828543

#Model with K=10 more profitable than K=1

######################
#   Model 5b        #
######################
#Set K=100 for KNN model 5b
set.seed(1)
model.knn5b=knn(x.train.std,x.valid.std,c.train,k=100)
mean(c.valid != model.knn5b)
# [1] 0.2215064

table(model.knn5b ,c.valid)
#             c.valid
#model.knn5b   0   1
#           0 600  28
#           1 419 971
#419+971=1390
#14.5*971-2*1390=11299.5

profit.knn5b <- cumsum(14.5*c.valid[order(model.knn5b, decreasing=T)]-2)
plot(profit.knn5b) # see how profits change as more mailings are made

mean((as.numeric(as.character(model.knn5b)) - c.valid)^2)

#Model 5b is most profitable, but may overfit the data


#########################################
#    MODEL 6: DECISION TREE             #
#########################################

model.tree1 <- tree(as.factor(donr) ~ .,data=data.train.std.c)
plot(model.tree1)
text(model.tree1)

post.valid.tree0 <- predict(model.tree1,data.valid.std.c)
mean((as.numeric(as.character(post.valid.tree0[,2])) - c.valid)^2)
# [1] 0.1094977

profit.tree0 <- cumsum(14.5*c.valid[order(post.valid.tree0[,2], decreasing=T)]-2)
plot(profit.tree0) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.tree0)
# [1] 1362

cutoff.tree0 <- sort(post.valid.tree0[,2], decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.tree0 <- ifelse(post.valid.tree0[,2] > cutoff.tree0, 1, 0) # mail to everyone above the cutoff
table(chat.valid.tree0, c.valid) # classification table

# c.valid
# chat.valid.tree0   0   1
#                0 667  32
#                1 352 967

# check n.mail.valid = 352+967 = 1319
##I have noticed that with trees, since groups of observations are less than a given point
#### this won't align with the actual n.mail.valid
# check profit = 14.5*967-2*1319 = 11383.5

model.tree1.cv =cv.tree(model.tree1 ,FUN=prune.misclass )

par(mfrow=c(1,2))
plot(model.tree1.cv$size ,model.tree1.cv$dev ,type="b")
plot(model.tree1.cv$k ,model.tree1.cv$dev ,type="b")

model.tree1.prune =prune.misclass(model.tree1,best =10)
plot(model.tree1.prune)
text(model.tree1.prune)

post.valid.tree1 <- predict(model.tree1.prune,data.valid.std.c)

mean((as.numeric(as.character(post.valid.tree1[,2])) - c.valid)^2)
# [1] 0.110671


profit.tree1 <- cumsum(14.5*c.valid[order(post.valid.tree1[,2], decreasing=T)]-2)
plot(profit.tree0) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.tree1)

cutoff.tree1 <- sort(post.valid.tree1[,2], decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.tree1 <- ifelse(post.valid.tree1[,2] > cutoff.tree1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.tree1, c.valid) # classification table

# c.valid
# chat.valid.tree0   0   1
#                0 667  32
#                1 352 967

# check n.mail.valid = 352+967 = 1319
# check profit = 14.5*967-2*1319 = 11383.5

##It's all the same... 

#########################################
#    MODEL 7: BOOSTS & RANDOM FOREST    #
#########################################

model.boost1 =gbm(donr~.,data=data.train.std.c, distribution="gaussian",n.trees =5000 , interaction.depth =4,shrinkage =0.2,
                  verbose =F)

post.valid.boost1 = predict(model.boost1,newdata = data.valid.std.c,n.trees =5000)

mean((post.valid.boost1 - c.valid)^2)
# [1] 0.1214981

model.RF1 <- randomForest(as.factor(donr)~.,data=data.train.std.c ,
                          mtry=13, ntree =25)

post.valid.RF1 = predict(model.RF1,newdata = data.valid.std.c)

mean((as.numeric(as.character(post.valid.RF1)) - c.valid)^2)
# [1]  0.1154609

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.RF1 <- cumsum(14.5*c.valid[order(post.valid.RF1, decreasing=T)]-2)
plot(profit.RF1) # see how profits change as more mailings are made


table(post.valid.RF1,c.valid)
# c.valid
# post.valid.RF1   0   1
# 0 887 101
# 1 132 898


# check n.mail.valid = 132+898 = 1030
# check profit = 14.5*898-2*1030 = 10961

#########################################
#    MODEL 8: Support Vector Machine    #
#########################################

library(doParallel)
cl <- makeCluster(detectCores()) 
registerDoParallel(cl)

model.svm =svm(donr~., data=data.train.std.c, kernel ="linear", cost =1e5)
summary(model.svm)
plot(model.svm , data.train.std.c$donr)

post.valid.svm =predict(model.svm ,data.valid.std.c)

profit.svm <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm) # see how profits change as more mailings are made

n.mail.valid <- which.max(profit.svm)
c(n.mail.valid, max(profit.svm))

cutoff.svm <- sort(post.valid.svm, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm <- ifelse(post.valid.svm > cutoff.svm, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm, c.valid) # classification table

#                c.valid
# chat.valid.svm   0   1
#              0  86   7
#              1 933 992

# check n.mail.valid = 933+992 = 1925
# check profit = 14.5*992-2*1925 = 10534

set.seed(1)
svm.tune=tune(svm,donr~.,data=data.train.std.c ,kernel ="radial",ranges =list(cost=c(0.001 , 0.01, 0.1, 1,5,10,100) ))
summary(svm.tune)

bestmod =svm.tune$best.model
summary(bestmod)

post.valid.svm =predict(bestmod,data.valid.std.c)

profit.svm <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm) # see how profits change as more mailings are made

n.mail.valid <- which.max(profit.svm)
c(n.mail.valid, max(profit.svm))
# 1444.0 11336.5

cutoff.svm <- sort(post.valid.svm, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm <- ifelse(post.valid.svm > cutoff.svm, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm, c.valid) # classification table
# 
# c.valid
# chat.valid.svm   0   1
#              0 556  18
#              1 463 981

# Results

# n.mail Profit  Model
# 1391   11631   LDA1
# 1343   11640   Log1
# 1446   11419.5 GAM1
# 1369   11341.5 QDA
# 1126   9913.5  KNN
# 1319   11383.5 Unaltered Tree
# 1030   10961   RF
# 1925   10534   Linear SVM (untuned)
# 1444   11336.5 Radial SVM (tuned)

##Logistic is the best!  Most profit.

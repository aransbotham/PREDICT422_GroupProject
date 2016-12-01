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
                      ,"lazyeval"
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
                      ,"randomForest"
                      ,"doParallel"
                      ,"iterators"
                      ,"foreach"
                      ,"parallel")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]
if(length(new.packages)) install.packages(new.packages)

#Load all packages
lapply(list.of.packages, require, character.only = TRUE)

# Load the diabetes data
data <- read.csv(file="charity.csv",stringsAsFactors=FALSE,header=TRUE,quote="",comment.char="")

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
#Some variables are bimodal

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

##Results:
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

##Visualize correlations
corrgram(charity.t, order=TRUE, lower.panel=panel.shade,
         upper.panel=panel.pie, text.panel=panel.txt,
         main="Correlations")

corrplot(M, method = "square") #plot matrix

##If we install pls earlier, then the corrplot function doesn't work properly.
library(pls)

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
#    MODEL 1: Logistic: Full model      #
#########################################
set.seed(1)
model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))


post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid1 <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid1, max(profit.log1)) # report number of mailings and maximum profit
#[1]  1397 11387

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid1+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table
#                 c.valid
# chat.valid.log1   0   1
#               0 600  21
#               1 419 978
# check n.mail.valid = 419+978 = 1397
# check profit = 14.5*978-2*1397 = 11387

##########################################
# Model 1a: Logistic w/ significant only #        
##########################################

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

#Resulting model is not more profitable than Full Model


##########################################
# Model 1b: Logistic backward selection  #        
##########################################
#Try logistic model with additional terms & backward selection

set.seed(1)
model.log1b <- glm(donr~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + I(wrat^2)
                   + avhv + incm + I(incm^2) + inca + I(inca^2) + plow + npro + tgif + I(tgif^2) + lgif +I(lgif^2)
                   + rgif + I(rgif^2) + tdon + tlag + agif + I(agif^2),data.train.std.c, family=binomial("logit"))

post.valid.log1b <- predict(model.log1b, data.valid.std.c, type="response") # n.valid post probs

profit.log1b <- cumsum(14.5*c.valid[order(post.valid.log1b, decreasing=T)]-2)
plot(profit.log1b) # see how profits change as more mailings are made
n.mail.valid1b <- which.max(profit.log1b) # number of mailings that maximizes profits
c(n.mail.valid1b, max(profit.log1b)) # report number of mailings and maximum profit
#[1]  1330 11637

cutoff.log1b <- sort(post.valid.log1b, decreasing=T)[n.mail.valid1b+1] # set cutoff based on n.mail.valid
chat.valid.log1b <- ifelse(post.valid.log1b>cutoff.log1b, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1b, c.valid) # classification table

#                  c.valid
# chat.valid.log1b   0   1
#                0 675  13
#                1 344 986

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
#[1]  1302.0 11649.5

cutoff.log1b_r1 <- sort(post.valid.log1b_r1, decreasing=T)[n.mail.valid1b_r1+1] # set cutoff based on n.mail.valid
chat.valid.log1b_r1 <- ifelse(post.valid.log1b_r1>cutoff.log1b, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1b_r1, c.valid) # classification table

#                     c.valid
# chat.valid.log1b_r1   0   1
#                   0 670  14
#                   1 349 985

#20 variable logistic model (model.log1b_r1) is most profitable . 

##Log Odds
exp(coef(model.log1b_r1))
# (Intercept)        reg1        reg2        home        chld   I(hinc^2)        genf        wrat   I(wrat^2) 
# 2.50563242  1.92965236  4.47780623  4.01570777  0.09033011  0.33427699  0.91727758  1.63353273  0.65764229 
# incm        inca   I(inca^2)        plow        tgif   I(tgif^2)        lgif   I(rgif^2)        tdon 
# 1.71730390  1.12595586  1.00213206  0.93293112  1.89711195  0.93196868  0.79267906  1.03948266  0.73711746 
# tlag        agif 
# 0.57772531  1.17439419 
exp(cbind(OR = coef(model.log1b_r1), confint(model.log1b_r1)))

#Odds Ratio
#               OR        2.5 %     97.5 %
# (Intercept) 2.50563242 1.99943688 3.1502925
# reg1        1.92965236 1.71062680 2.1824684
# reg2        4.47780623 3.89212895 5.1787041
# home        4.01570777 3.41327149 4.7613630
# chld        0.09033011 0.07583776 0.1067378
# I(hinc^2)   0.33427699 0.30007289 0.3708639
# genf        0.91727758 0.82104858 1.0242978
# wrat        1.63353273 1.34499428 1.9871924
# I(wrat^2)   0.65764229 0.57529258 0.7486201
# incm        1.71730390 1.35146914 2.1857712
# inca        1.12595586 0.91453371 1.3864517
# I(inca^2)   1.00213206 0.91869506 1.0944322
# plow        0.93293112 0.74903602 1.1613059
# tgif        1.89711195 1.68300253 2.1430667
# I(tgif^2)   0.93196868 0.85856296 1.0085868
# lgif        0.79267906 0.65051949 0.9657211
# I(rgif^2)   1.03948266 0.95557675 1.1323442
# tdon        0.73711746 0.65200155 0.8318903
# tlag        0.57772531 0.51221950 0.6500910
# agif        1.17439419 0.96413403 1.4314516

#########################################
#    MODEL 2: Logistic GAM              #
#########################################
set.seed(1)
model.gam1=gam(donr~reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
             avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + s(agif ,df =5),
           family = binomial , data = data.train.std.c)

post.valid.gam1 <- predict(model.gam1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.gam1 <- cumsum(14.5*c.valid[order(post.valid.gam1, decreasing=T)]-2)
plot(profit.gam1) # see how profits change as more mailings are made
n.mail.valid1 <- which.max(profit.gam1) # number of mailings that maximizes profits
c(n.mail.valid1, max(profit.gam1)) # report number of mailings and maximum profit
# 1396 11389

cutoff.gam1 <- sort(post.valid.gam1, decreasing=T)[n.mail.valid1+1] # set cutoff based on n.mail.valid
chat.valid.gam1 <- ifelse(post.valid.gam1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.gam1, c.valid) # classification table
#                 c.valid
# chat.valid.gam1   0   1
#               0 597  21
# 1               422 978
# check n.mail.valid = 422+978 = 1400
# check profit = 14.5*978-2*1400 = 11381 # This doesn't equal 11389...

##########################################
#    MODEL 2a: Logistic GAM w/ best vars #
##########################################

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
set.seed(1)
model.lda1 <- lda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c) 

# Note: strictly speaking, LDA should not be used with qualitative predictors,
# but in practice it often is if the goal is simply to find a good predictive model

post.valid.lda1 <- predict(model.lda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.lda1 <- cumsum(14.5*c.valid[order(post.valid.lda1, decreasing=T)]-2)
plot(profit.lda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.lda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.lda1)) # report number of mailings and maximum profit
# 1363.0 11643.5

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table
#                 c.valid
# chat.valid.lda1   0   1
#               0 647   8
#               1 372 991
# check n.mail.valid = 372 + 991 = 1363
# check profit = 14.5*991-2*1363 = 11643.5


#########################################
#    MODEL 3a: LDA with subset of vars  #
#########################################
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
# 1336.0 11639.5

cutoff.lda3a <- sort(post.valid.lda3a, decreasing=T)[n.mail.valid3a+1] # set cutoff based on n.mail.valid
chat.valid.lda3a <- ifelse(post.valid.lda3a>cutoff.lda3a, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda3a, c.valid)
#                 c.valid
#chat.valid.lda3a   0   1
#               0 670  12
#               1 349 987
#349+987=1336
#14.5*987-2*1336=11639.5

#Model 3 marginally more profitable than Model 3a

##Plot LDA?
png("./plots", width=10000, height=10000, pointsize=12)
partimat(as.factor(donr) ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
           +     avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,data=data.train.std.c,method="lda")


#########################################
#    MODEL 4: QDA                       #
#########################################

model.qda1 =qda(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
              avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif,data= data.train.std.c)

post.valid.qda1 <- predict(model.qda1, data.valid.std.c)$posterior[,2] # n.valid.c post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.qda1 <- cumsum(14.5*c.valid[order(post.valid.qda1, decreasing=T)]-2)
plot(profit.qda1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.qda1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.qda1)) # report number of mailings and maximum profit
# 1396.0 11229.5

cutoff.qda1 <- sort(post.valid.qda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.qda1 <- ifelse(post.valid.qda1>cutoff.qda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.qda1, c.valid) # classification table
#                 c.valid
# chat.valid.qda1   0   1
#               0 590  32
#               1 429 967
# check n.mail.valid = 429+967 = 1396
# check profit = 14.5*967-2*1396 = 11229.5


#########################################
#    MODEL 4: QDA with subset of vars   #
#########################################

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
# [1] 0.2205154

table(model.knn1 ,c.valid)
#            c.valid
# model.knn1   0   1
#          0 738 164
#          1 281 835

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.knn1 <- cumsum(14.5*c.valid[order(model.knn1, decreasing=T)]-2)
plot(profit.knn1) # see how profits change as more mailings are made

mean((as.numeric(as.character(model.knn1)) - c.valid)^2)
# [1] 0.2205154

# check n.mail.valid = 281+835 = 1116
# check profit = 14.5*835-2*1116 = 9875.5


#########################################
#    MODEL 5a: KNN                      #
#########################################
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

# check n.mail.valid = 310+940 = 1250
# check profit = 14.5*940-2*1250 = 11130


#Model with K=10 more profitable than K=1

#########################################
#    MODEL 5b: KNN                      #
#########################################
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
# [1] 0.2215064

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
# [1] 1362 11413.5

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
# [1] 1362 11413.5

cutoff.tree1 <- sort(post.valid.tree1[,2], decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.tree1 <- ifelse(post.valid.tree1[,2] > cutoff.tree1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.tree1, c.valid) # classification table

# c.valid
# chat.valid.tree0   0   1
#                0 667  32
#                1 352 967

# check n.mail.valid = 352+967 = 1319
# check profit = 14.5*967-2*1319 = 11383.5

##Pruned and unpruned tree are the same

#########################################
#    MODEL 6a: DECISION TREE w/ subset  #
#########################################

model.tree2 <- tree(as.factor(donr) ~ reg1 + reg2 + home + chld + I(hinc^2) + genf + wrat + I(wrat^2) + incm + inca + 
                      I(inca^2) + plow + tgif + I(tgif^2) + lgif + I(rgif^2) + tdon + tlag + agif
                    ,data=data.train.std.c)
plot(model.tree2)
text(model.tree2)

post.valid.tree0 <- predict(model.tree2,data.valid.std.c)
mean((as.numeric(as.character(post.valid.tree0[,2])) - c.valid)^2)
# [1] 0.1072104

profit.tree0 <- cumsum(14.5*c.valid[order(post.valid.tree0[,2], decreasing=T)]-2)
plot(profit.tree0) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.tree0)
# [1]  1487 11381

cutoff.tree0 <- sort(post.valid.tree0[,2], decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.tree0 <- ifelse(post.valid.tree0[,2] > cutoff.tree0, 1, 0) # mail to everyone above the cutoff
table(chat.valid.tree0, c.valid) # classification table
# 
#                  c.valid
# chat.valid.tree0   0   1
#                0 617  28
#                1 402 971

# check n.mail.valid = 402+971 = 1373

##I have noticed that with trees, since groups of observations are less than a given point
#### this won't align with the actual n.mail.valid
# check profit = 14.5*971-2*1373 = 11333.5

##Tree using all variables performs better than tree using the subset

#########################################
#    MODEL 7: BOOSTS & RANDOM FOREST    #
#########################################

set.seed(1)
model.RF1 <- randomForest(as.factor(donr)~.,data=data.train.std.c ,
             mtry=13, ntree =25)

post.valid.RF1 = predict(model.RF1,newdata = data.valid.std.c)

mean((as.numeric(as.character(post.valid.RF1)) - c.valid)^2)
#[1] 0.1149653

varImpPlot(model.RF1)

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.RF1 <- cumsum(14.5*c.valid[order(post.valid.RF1, decreasing=T)]-2)
plot(profit.RF1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.RF1)
# 1040  11028
table(post.valid.RF1,c.valid)
#                c.valid
# post.valid.RF1   0   1
# 0              883  95
# 1              136 904

# check n.mail.valid = 136+904 = 1040
# check profit = 14.5*904-2*1040 = 11028

set.seed(1)
model.boost1 =gbm(donr~.,data=data.train.std.c, distribution="gaussian",n.trees =5000 , interaction.depth =4,shrinkage =0.2,
                  verbose =F)

#produce plot of relative influence
summary.gbm(model.boost1)
#var    rel.inf
#chld chld 16.5050114
#agif agif  9.0830404
#avhv avhv  7.7627473
#tgif tgif  7.6615290
#hinc hinc  6.6249256
#npro npro  6.6128775
#incm incm  6.1582047
#inca inca  5.4056824
#tdon tdon  4.4863957
#plow plow  4.3549452
#lgif lgif  4.3471496
#wrat wrat  4.2586813
#reg2 reg2  4.1002435
#rgif rgif  3.9026428
#home home  3.5211727
#tlag tlag  3.2892911
#reg1 reg1  0.9795926
#genf genf  0.4274159
#reg3 reg3  0.2622024
#reg4 reg4  0.2562489

post.valid.boost1 = predict(model.boost1,newdata = data.valid.std.c,n.trees =5000)

mean((post.valid.boost1 - c.valid)^2)
# [1] 0.1198942

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.boost1 <- cumsum(14.5*c.valid[order(post.valid.boost1, decreasing=T)]-2)
plot(profit.boost1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.boost1)
# 1344 11594.5

cutoff.boost1 <- sort(post.valid.boost1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.boost1 <- ifelse(post.valid.boost1 > cutoff.boost1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.boost1, c.valid) # classification table

#                   c.valid
# chat.valid.boost1   0   1
#                 0 660  14
#                 1 359 985

# check n.mail.valid = 359+985 = 1344
# check profit = 14.5*985-2*1344 = 11594.5

#########################################
#    MODEL 8: Support Vector Machine    #
#########################################

library(doParallel)
cl <- makeCluster(detectCores()) 
registerDoParallel(cl)

set.seed(1)
model.svm =svm(donr~., data=data.train.std.c, kernel ="linear", cost =1e5)
summary(model.svm)
plot(model.svm , data.train.std.c$donr)

post.valid.svm =predict(model.svm ,data.valid.std.c)

profit.svm1 <- cumsum(14.5*c.valid[order(post.valid.svm, decreasing=T)]-2)
plot(profit.svm1) # see how profits change as more mailings are made

n.mail.valid <- which.max(profit.svm1)
c(n.mail.valid, max(profit.svm1))

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
# [1]  1366 11536

cutoff.svm <- sort(post.valid.svm, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.svm <- ifelse(post.valid.svm > cutoff.svm, 1, 0) # mail to everyone above the cutoff
table(chat.valid.svm, c.valid) # classification table
# 
               # c.valid
# chat.valid.svm   0   1
#              0 637  15
#              1 382 984

# Results

# n.mail Profit  Model
# 1397   11387   Log1
# 1583   11073   Log 1a
# 1302.0 11649.5 Log 1b
# 1396   11389   Log GAM1
# 1302.0 11649.5 Log GAM1a
# 1363   11642.5 LDA1
# 1336.0 11639.5 LDA1a
# 1396.0 11229.5 QDA
# 1421   11107   QDA1a
# 1116   9875.5  KNN
# 1250   11130   KNN1a
# 1390   11299.5 KNN1b
# 1362   11413.5 Unaltered Tree & Pruned Tree
# 1487   11381   Tree with subset vars
# 1308   11565   Boosted Tree
# 1055   11099.5 RF
# 1925   10534   Linear SVM (untuned)
# 1366   11536 Radial SVM (tuned)

##Logistic is the best!  Most profit.
## should use this model object for test set predictions: model.log1b_r1

#########################################################################################
#Question 3 -- Prediction Models for DAMT                                               #
#########################################################################################

#########################################
#    MODEL 1: Least Squares             #
#########################################

# Linear model with all variables.
model.lin1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                 data.train.std.y)
summary(model.lin1)

# Use regsubsets() to identify the best subset of predictor variables.
regfit.full <- regsubsets(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                            wrat + genf + avhv + incm + inca + plow + npro + tgif +
                            lgif + rgif + tdon + tlag + agif, data.train.std.y, nvmax = 20)

reg.summary = summary(regfit.full)
names(reg.summary)
reg.summary$rsq
reg.summary$adjr2

which.max(reg.summary$adjr2) #15
which.min(reg.summary$cp) #14
which.min(reg.summary$bic) #11

plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")

# Start with model with highest adjr2. 
coef(regfit.full, 15)
model.lin2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf +
                   incm + plow + tgif + lgif + rgif + tdon + agif,
                 data.train.std.y)
summary(model.lin2)

# Run model on validation set.
model.lin2.valid <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf +
                         incm + plow + tgif + lgif + rgif + tdon + agif,
                       data.valid.std.y)
summary(model.lin2.valid)

# Make predictions for validation set based on model.
pred.valid.lin2 <- predict(model.lin2.valid, data.valid.std.y)

MPE_adjR <- mean((y.valid - pred.valid.lin2)^2)
StandardError_adjR <- sd((y.valid - pred.valid.lin2)^2)/sqrt(n.valid.y)

MPE_adjR # 1.50811
StandardError_adjR # 0.1570488

#### Now for the model with lowest cp. 
coef(regfit.full, 14)
model.lin_cp <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf +
                     incm + plow + tgif + lgif + rgif + tdon + agif,
                   data.train.std.y)
summary(model.lin_cp)

# Run model on validation set.
model.lin_cp.valid <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf +
                           incm + plow + tgif + lgif + rgif + tdon + agif,
                         data.valid.std.y)
summary(model.lin_cp.valid)

# Make predictions for validation set based on model.
pred.valid.lin_cp <- predict(model.lin_cp.valid, data.valid.std.y)

MPE_cp <- mean((y.valid - pred.valid.lin_cp)^2)
StandardError_cp <- sd((y.valid - pred.valid.lin_cp)^2)/sqrt(n.valid.y)

MPE_cp # 1.50853
StandardError_cp # 0.1568465

#### Now for the model with the lowest bic
coef(regfit.full, 11)
model.lin_bic <- lm(damt ~ reg3 + reg4 + home + chld + hinc +
                      incm + plow + tgif + lgif + rgif + agif,
                    data.train.std.y)
summary(model.lin_bic)

# Run model on validation set.
model.lin_bic.valid <- lm(damt ~ reg3 + reg4 + home + chld + hinc +
                            incm + plow + tgif + lgif + rgif + agif,
                          data.valid.std.y)
summary(model.lin_bic.valid)

# Make predictions for validation set based on model.
pred.valid.lin_bic <- predict(model.lin_bic.valid, data.valid.std.y)

MPE_bic <- mean((y.valid - pred.valid.lin_bic)^2)
StandardError_bic <- sd((y.valid - pred.valid.lin_bic)^2)/sqrt(n.valid.y)

MPE_bic # 1.527695
StandardError_bic # 0.1596775


##
MPE_adjR # 1.50811
StandardError_adjR # 0.1570488

MPE_cp # 1.50853
StandardError_cp # 0.1568465

MPE_bic # 1.527695
StandardError_bic # 0.1596775

#
# Model using highest Adjusted R-squared is best.

#########################################
#    MODEL 2: Best Subset w/ k-fold cv  #
#########################################
set.seed(1)
regfit.best <- regsubsets(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                            wrat + genf + avhv + incm + inca + plow + npro + tgif +
                            lgif + rgif + tdon + tlag + agif, data.train.std.y, nvmax = 20)

set.seed(1)
test.mat = model.matrix(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                          wrat + genf + avhv + incm + inca + plow + npro + tgif +
                          lgif + rgif + tdon + tlag + agif, data.valid.std.y)

val.errors <- rep(NA,20)
for (i in 1:20) {
  coefi = coef(regfit.best, id=i)
  pred = test.mat[,names(coefi)]%*%coefi
  val.errors[i] = mean((y.valid - pred)^2)}
val.errors
which.min(val.errors) # 18
coef(regfit.best,18)
  
# create plot
par(mfrow =c(1,1))
plot(val.errors, xlab="Number of Variables", type='b')
points (18,val.errors[18], col = "red",cex = 1, pch = 20)

model.lin4 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + wrat +
                   genf + incm + inca + plow + tgif + lgif + rgif + tdon + tlag + agif, 
                 data.train.std.y)
summary(model.lin4)

# Run model on validation set.
model.lin4.valid <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + wrat +
                         genf + incm + inca + plow + tgif + lgif + rgif + tdon + tlag + agif, 
                       data.valid.std.y)
summary(model.lin4.valid)

# Make predictions for validation set based on model.
pred.valid.lin4 <- predict(model.lin4, data.valid.std.y)
MPE4 <- mean((y.valid - pred.valid.lin4)^2)
StandardError4 <- sd((y.valid - pred.valid.lin4)^2)/sqrt(n.valid.y)
MPE4 # 1.555443
StandardError4 # 0.1611711
  
#########################################
#    MODEL 3: Principal Components       #
#########################################

set.seed(1)
pcr.fit = pcr(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                wrat + genf + avhv + incm + inca + plow + npro + tgif +
                lgif + rgif + tdon + tlag + agif, data = data.train.std.y,
              scale = TRUE, validation = "CV")

summary(pcr.fit)
  
validationplot(pcr.fit, val.type = "MSEP", type = "b", pch=20, col="black")

# There is an drop in the graph at 5, and the lowest point is around 20.
# The drop at 5 makes me think five components may be enough.

set.seed(1)
pcr.pred = predict(pcr.fit, data.valid.std.y, ncomp=5)

MPE5 <- mean((y.valid - pcr.pred)^2)
StandardError5 <- sd((y.valid - pcr.pred)^2)/sqrt(n.valid.y)

MPE5 # 1.812705
StandardError5 # 0.1688532

#Use 15 components
set.seed(1)
pcr.pred2 = predict(pcr.fit, data.valid.std.y, ncomp=15)

MPE6 <- mean((y.valid - pcr.pred2)^2)
MPE6 # 1.598671

StandardError6 <- sd((y.valid - pcr.pred2)^2)/sqrt(n.valid.y)
StandardError6 # 0.1607489

# Use 20 components
set.seed(1)
pcr.pred2 = predict(pcr.fit, data.valid.std.y, ncomp=20)

MPE7 <- mean((y.valid - pcr.pred2)^2)
MPE7 # 1.556378

StandardError7 <- sd((y.valid - pcr.pred2)^2)/sqrt(n.valid.y)
StandardError7 # 0.161215

##
# 5 components
MPE5 # 1.812705
StandardError5 # 0.1688532
# 15 components
MPE6 # 1.598671
StandardError6 # 0.1607489
# 20 components
MPE7 # 1.556378
StandardError7 # 0.161215

# 20 components is best.  
  
#########################################
#    MODEL 4: Partial Least Squares     #
#########################################

set.seed(1)
pls.fit = plsr(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                 wrat + genf + avhv + incm + inca + plow + npro + tgif +
                 lgif + rgif + tdon + tlag + agif, data = data.train.std.y,
               scale = TRUE, validation = "CV") 

summary(pls.fit)

validationplot(pls.fit, val.type="MSEP", type = "b")
# There is an drop in the graph at 3, with minimal reduction after.
# The drop at 3 makes me think three components is enough.

set.seed(1)
pls.pred = predict(pls.fit, data.valid.std.y, ncomp=3)

MPE8 <- mean((y.valid - pls.pred)^2)
StandardError8 <- sd((y.valid - pls.pred)^2)/sqrt(n.valid.y)

MPE8 # 1.592151
StandardError8 # 0.1613484  

  
#########################################
#    MODEL 5: Ridge Regression          #
#########################################

set.seed(1)

grid = 10^seq(10, -2, length=100)
mat.train <- data.matrix(data.train.std.y)
mat.train <- mat.train[,-21]
# Remove damt to so that the response is not on both sides of equation.

ridge.mod = glmnet(mat.train, y.train, alpha=0, lambda=grid,
                   thresh=1e-12)

# Use cross-validation to choose lambda.
set.seed(1)
cv.out = cv.glmnet(mat.train, y.train, alpha=0)
plot(cv.out)

bestlam = cv.out$lambda.min
bestlam # 0.1252296

# Make predictions and compute errors.
mat.valid = as.matrix(data.valid.std.y)
mat.valid <- mat.valid[,-21]

set.seed(1)
ridge.pred = predict(ridge.mod, s=bestlam, newx=mat.valid)

MPE9 <- mean((y.valid - ridge.pred)^2)
StandardError9 <- sd((y.valid - ridge.pred)^2)/sqrt(n.valid.y)

MPE9 # 1.572113
StandardError9 # 0.1627705
  
ridge.coef= coef(ridge.mod,s=bestlam)
# (Intercept) 14.22344033
# reg1        -0.06495925
# reg2        -0.12240428
# reg3         0.28014764
# reg4         0.58541542
# home         0.20548728
# chld        -0.54971631
# hinc         0.47980195
# genf        -0.05763863
# wrat         0.01231801
# avhv        -0.01751383
# incm         0.30239430
# inca         0.06359597
# plow         0.30357092
# npro         0.02575266
# tgif         0.16534000
# lgif         0.39461843
# rgif         0.44561459
# tdon         0.07087720
# tlag         0.02834013
# agif         0.38205327
#########################################
#    MODEL 6: Lasso Regression          #
#########################################

set.seed(1)

# Use matrices and lambda grid created for ridge regression.
lasso.mod = glmnet(mat.train, y.train, alpha=1, lambda=grid)

# Use cross-validation to select lambda.
set.seed(1)
cv.out.lasso = cv.glmnet(mat.train, y.train, alpha=1)
plot(cv.out.lasso)

bestlamlasso = cv.out.lasso$lambda.min
bestlamlasso # 0.005174504

set.seed(1)
lasso.pred = predict(lasso.mod, s=bestlamlasso, newx=mat.valid)

MPE10 <- mean((y.valid - lasso.pred)^2)
StandardError10 <- sd((y.valid - lasso.pred)^2)/sqrt(n.valid.y)

MPE10 # 1.562046
StandardError10 # 0.1611463

lasso.coef= coef(lasso.mod,s=bestlam)
# (Intercept) 14.338428658
# reg1         .          
# reg2        -0.051092397
# reg3         0.203743788
# reg4         0.525828946
# home         .          
# chld        -0.430556714
# hinc         0.338995574
# genf         .          
# wrat         .          
# avhv         .          
# incm         0.005835249
# inca         .          
# plow         .          
# npro         .          
# tgif         0.076711998
# lgif         0.380005233
# rgif         0.399168041
# tdon         .          
# tlag         .          
# agif         0.332079318
#########################################
# FINAL RESULTS                         #
#########################################
# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

class.model <- model.log1b_r1
cutoff <- cutoff.log1b_r1 
reg.model <- model.lin_cp

data.test.std$c.hat.prob <- predict(class.model, data.test.std, type="response")
data.test.std$c.hat.class <- ifelse(data.test.std$c.hat.prob > cutoff,1,0)

data.test.std$y.hat <- ifelse(data.test.std$c.hat.class == 1,predict(reg.model,data.test.std),"-")

# n.valid post probs
ip <- data.frame(chat=data.test.std$c.hat.class, yhat=data.test.std$y.hat) # data frame with two variables: chat and yhat
write.csv(ip, file="BFSZ.csv", row.names=FALSE) # use your initials for the file name

# submit the csv file in Canvas for evaluation based on actual test donr and damt values

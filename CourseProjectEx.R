# PREDICT 422 Practical Machine Learning

# Course Project - Example R Script File

# OBJECTIVE: A charitable organization wishes to develop a machine learning
# model to improve the cost-effectiveness of their direct marketing campaigns
# to previous donors.

# 1) Develop a classification model using data from the most recent campaign that
# can effectively capture likely donors so that the expected net profit is maximized.

# 2) Develop a prediction model to predict donation amounts for donors - the data
# for this will consist of the records for donors only.

# load the data
charity <- read.csv(file.choose()) # load the "charity.csv" file

# predictor transformations

charity.t <- charity
charity.t$avhv <- log(charity.t$avhv)
# add further transformations if desired
# for example, some statistical methods can struggle when predictors are highly skewed

# set up data for analysis

data.train <- charity.t[charity$part=="train",]
x.train <- data.train[,2:21]
c.train <- data.train[,22] # donr
n.train.c <- length(c.train) # 3984
y.train <- data.train[c.train==1,23] # damt for observations with donr=1
n.train.y <- length(y.train) # 1995

data.valid <- charity.t[charity$part=="valid",]
x.valid <- data.valid[,2:21]
c.valid <- data.valid[,22] # donr
n.valid.c <- length(c.valid) # 2018
y.valid <- data.valid[c.valid==1,23] # damt for observations with donr=1
n.valid.y <- length(y.valid) # 999

data.test <- charity.t[charity$part=="test",]
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


##### CLASSIFICATION MODELING ######

# linear discriminant analysis

library(MASS)

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
# 1329.0 11624.5

cutoff.lda1 <- sort(post.valid.lda1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.lda1 <- ifelse(post.valid.lda1>cutoff.lda1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.lda1, c.valid) # classification table
#               c.valid
#chat.valid.lda1   0   1
#              0 675  14
#              1 344 985
# check n.mail.valid = 344+985 = 1329
# check profit = 14.5*985-2*1329 = 11624.5

# logistic regression

model.log1 <- glm(donr ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + I(hinc^2) + genf + wrat + 
                    avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                  data.train.std.c, family=binomial("logit"))

post.valid.log1 <- predict(model.log1, data.valid.std.c, type="response") # n.valid post probs

# calculate ordered profit function using average donation = $14.50 and mailing cost = $2

profit.log1 <- cumsum(14.5*c.valid[order(post.valid.log1, decreasing=T)]-2)
plot(profit.log1) # see how profits change as more mailings are made
n.mail.valid <- which.max(profit.log1) # number of mailings that maximizes profits
c(n.mail.valid, max(profit.log1)) # report number of mailings and maximum profit
# 1291.0 11642.5

cutoff.log1 <- sort(post.valid.log1, decreasing=T)[n.mail.valid+1] # set cutoff based on n.mail.valid
chat.valid.log1 <- ifelse(post.valid.log1>cutoff.log1, 1, 0) # mail to everyone above the cutoff
table(chat.valid.log1, c.valid) # classification table
#               c.valid
#chat.valid.log1   0   1
#              0 709  18
#              1 310 981
# check n.mail.valid = 310+981 = 1291
# check profit = 14.5*981-2*1291 = 11642.5

# Results

# n.mail Profit  Model
# 1329   11624.5 LDA1
# 1291   11642.5 Log1

# select model.log1 since it has maximum profit in the validation sample

post.test <- predict(model.log1, data.test.std, type="response") # post probs for test data

# Oversampling adjustment for calculating number of mailings for test set

n.mail.valid <- which.max(profit.log1)
tr.rate <- .1 # typical response rate is .1
vr.rate <- .5 # whereas validation response rate is .5
adj.test.1 <- (n.mail.valid/n.valid.c)/(vr.rate/tr.rate) # adjustment for mail yes
adj.test.0 <- ((n.valid.c-n.mail.valid)/n.valid.c)/((1-vr.rate)/(1-tr.rate)) # adjustment for mail no
adj.test <- adj.test.1/(adj.test.1+adj.test.0) # scale into a proportion
n.mail.test <- round(n.test*adj.test, 0) # calculate number of mailings for test set

cutoff.test <- sort(post.test, decreasing=T)[n.mail.test+1] # set cutoff based on n.mail.test
chat.test <- ifelse(post.test>cutoff.test, 1, 0) # mail to everyone above the cutoff
table(chat.test)
#    0    1 
# 1676  331
# based on this model we'll mail to the 331 highest posterior probabilities

# See below for saving chat.test into a file for submission



##### PREDICTION MODELING ######

# Least squares regression

model.ls1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ls1 <- predict(model.ls1, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls1)^2) # mean prediction error
# 1.867523
sd((y.valid - pred.valid.ls1)^2)/sqrt(n.valid.y) # std error
# 0.1696615

# drop wrat for illustrative purposes
model.ls2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + 
                  avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                data.train.std.y)

pred.valid.ls2 <- predict(model.ls2, newdata = data.valid.std.y) # validation predictions
mean((y.valid - pred.valid.ls2)^2) # mean prediction error
# 1.867433
sd((y.valid - pred.valid.ls2)^2)/sqrt(n.valid.y) # std error
# 0.1696498

# Results

# MPE  Model
# 1.867523 LS1
# 1.867433 LS2

# select model.ls2 since it has minimum mean prediction error in the validation sample

yhat.test <- predict(model.ls2, newdata = data.test.std) # test predictions




# FINAL RESULTS

# Save final results for both classification and regression

length(chat.test) # check length = 2007
length(yhat.test) # check length = 2007
chat.test[1:10] # check this consists of 0s and 1s
yhat.test[1:10] # check this consists of plausible predictions of damt

ip <- data.frame(chat=chat.test, yhat=yhat.test) # data frame with two variables: chat and yhat
write.csv(ip, file="ABC.csv", row.names=FALSE) # use your initials for the file name

# submit the csv file in Canvas for evaluation based on actual test donr and damt values


#### 3. Develop a prediction model for the DAMT variable
####MODEL ONE: LEAST SQUARES REGRESSION
# Linear model with all variables.
model.lin1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                   avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                 data.train.std.y)
summary(model.lin1)

# Use regsubsets() to identify the best subset of predictor variables.
library(leaps)
regfit.full <- regsubsets(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                            wrat + genf + avhv + incm + inca + plow + npro + tgif +
                            lgif + rgif + tdon + tlag + agif, data.train.std.y, nvmax = 20)
reg.summary = summary(regfit.full)
names(reg.summary)
reg.summary$rsq
reg.summary$adjr2
which.max(reg.summary$adjr2) #16
which.min(reg.summary$cp) #13
which.min(reg.summary$bic) #10
plot(regfit.full, scale = "Cp")
plot(regfit.full, scale = "bic")

# Start with model with highest adjr2. 
coef(regfit.full, 16)
model.lin2 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf +
                   incm + inca + plow + tgif + lgif + rgif + tdon + agif,
                 data.train.std.y)
summary(model.lin2)

# Run model on validation set.
model.lin2.valid <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf +
                         incm + inca + plow + tgif + lgif + rgif + tdon + agif,
                       data.valid.std.y)
summary(model.lin2.valid)

# Make predictions for validation set based on model.
pred.valid.lin2 <- predict(model.lin2.valid, data.valid.std.y)
MPE2 <- mean((y.valid - pred.valid.lin2)^2)
StandardError2 <- sd((y.valid - pred.valid.lin2)^2)/sqrt(n.valid.y)
MPE2 #1.7971
StandardError2 #0.1685

#### Now for the model with lowest cp. 
coef(regfit.full, 13)
model.lin_cp <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf +
                   incm + plow + npro + rgif + tdon + agif,
                 data.train.std.y)
summary(model.lin_cp)

# Run model on validation set.
model.lin_cp.valid <- lm(damt ~ reg2 + reg3 + reg4 + home + chld + hinc + genf +
                           incm + plow + npro + rgif + tdon + agif,
                       data.valid.std.y)
summary(model.lin_cp.valid)

# Make predictions for validation set based on model.
pred.valid.lin_cp <- predict(model.lin_cp.valid, data.valid.std.y)
MPE_cp <- mean((y.valid - pred.valid.lin_cp)^2)
StandardError_cp <- sd((y.valid - pred.valid.lin_cp)^2)/sqrt(n.valid.y)
MPE_cp #1.7912
StandardError_cp #0.1653

#### Now for the model with the lowest bic
coef(regfit.full, 10)
model.lin_bic <- lm(damt ~ reg3 + reg4 + home + chld + hinc +
                     incm + plow + npro + rgif + agif,
                   data.train.std.y)
summary(model.lin_bic)

# Run model on validation set.
model.lin_bic.valid <- lm(damt ~ reg3 + reg4 + home + chld + hinc +
                            incm + plow + npro + rgif + agif,
                         data.valid.std.y)
summary(model.lin_bic.valid)

# Make predictions for validation set based on model.
pred.valid.lin_bic <- predict(model.lin_bic.valid, data.valid.std.y)
MPE_bic <- mean((y.valid - pred.valid.lin_bic)^2)
StandardError_bic <- sd((y.valid - pred.valid.lin_bic)^2)/sqrt(n.valid.y)
MPE_bic #1.8127
StandardError_bic #0.1687

#
# Lowest CP is the model choosen based on stats above
# Include interaction terms and non-linear transformations--> **still need to do EDA
## I'll check the code to see what others have come up with

######################################################################
####MODEL TWO: BEST SUBSET WITH CROSS-VALIDATION
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
which.min(val.errors)
coef(regfit.best,10)

model.lin4 <- lm(damt ~ reg3 + reg4 + home + chld + hinc + incm
                  + plow + npro + rgif + agif, 
                 data.train.std.y)
summary(model.lin4)

# Run model on validation set.
model.lin4.valid <- lm(damt ~ reg3 + reg4 + home + chld + hinc + incm
                       + plow + npro + rgif + agif, 
                       data.valid.std.y)
summary(model.lin4.valid)

# Make predictions for validation set based on model.
pred.valid.lin4 <- predict(model.lin4, data.valid.std.y)
MPE4 <- mean((y.valid - pred.valid.lin4)^2)
StandardError4 <- sd((y.valid - pred.valid.lin4)^2)/sqrt(n.valid.y)
MPE4 #1.8579
StandardError4 #0.1694

#########################################################################
####MODEL THREE: PRINCIPLE COMPONENTS REGRESSION
library(pls)
set.seed(1)
pcr.fit = pcr(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
                wrat + genf + avhv + incm + inca + plow + npro + tgif +
                lgif + rgif + tdon + tlag + agif, data = data.train.std.y,
              scale = TRUE, validation = "CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP", type = "b")
# There is an drop in the graph at 5, and the lowest point is around 20.
# The drop at 5 makes me think five components may be enough.
set.seed(1)
pcr.pred = predict(pcr.fit, data.valid.std.y, ncomp=5)
MPE5 <- mean((y.valid - pcr.pred)^2)
StandardError5 <- sd((y.valid - pcr.pred)^2)/sqrt(n.valid.y)
MPE5 #2.1551
StandardError5 #0.1865
#Use 15 components
set.seed(1)
pcr.pred2 = predict(pcr.fit, data.valid.std.y, ncomp=15)
MPE6 <- mean((y.valid - pcr.pred2)^2)
MPE6 #1.8620
StandardError6 <- sd((y.valid - pcr.pred2)^2)/sqrt(n.valid.y)
StandardError6 #0.1692
# Use 20 components
set.seed(1)
pcr.pred2 = predict(pcr.fit, data.valid.std.y, ncomp=20)
MPE7 <- mean((y.valid - pcr.pred2)^2)
MPE7 #1.8667
StandardError7 <- sd((y.valid - pcr.pred2)^2)/sqrt(n.valid.y)
StandardError7 #0.1696

#I was wrong about 5 components but 15 is the best!

##########################################################################
####MODEL FOUR: PARTIAL LEAST SQUARES
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
StandardError7 <- sd((y.valid - pls.pred)^2)/sqrt(n.valid.y)
MPE8 #1.8760
StandardError8 #0.1716

###############################################################################
####MODEL FIVE: RIDGE REGRESSION
library(glmnet)
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
bestlam
# Make predictions and compute errors.
mat.valid = as.matrix(data.valid.std.y)
mat.valid <- mat.valid[,-21]
set.seed(1)
ridge.pred = predict(ridge.mod, s=bestlam, newx=mat.valid)
MPE9 <- mean((y.valid - ridge.pred)^2)
StandardError9 <- sd((y.valid - ridge.pred)^2)/sqrt(n.valid.y)
MPE9 #1.8717
StandardError9 #0.1710

##########################################################################
####MODEL SIX: LASSO
set.seed(1)
# Use matrices and lambda grid created for ridge regression.
lasso.mod = glmnet(mat.train, y.train, alpha=1, lambda=grid)
# Use cross-validation to select lambda.
set.seed(1)
cv.out.lasso = cv.glmnet(mat.train, y.train, alpha=1)
plot(cv.out.lasso)
bestlamlasso = cv.out.lasso$lambda.min
bestlamlasso
set.seed(1)
lasso.pred = predict(lasso.mod, s=bestlamlasso, newx=mat.valid)
MPE10 <- mean((y.valid - lasso.pred)^2)
StandardError10 <- sd((y.valid - lasso.pred)^2)/sqrt(n.valid.y)
MPE10 #1.8598
StandardError10 #0.1694

#should we do something with decision trees?

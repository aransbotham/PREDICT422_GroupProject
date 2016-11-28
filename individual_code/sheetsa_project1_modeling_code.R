
##Modeling Time!

#########################################################################################
#Question 1 -- Least squares regression model using all ten predictors (R function lm). #
#########################################################################################

model.lm <- lm(y~.,data = data.train)
lm.summary <- summary(model.lm)
lm.summary
# 
# Call:
#   lm(formula = y ~ ., data = data.train)
# 
# Residuals:
#   Min       1Q   Median       3Q      Max 
# -155.726  -36.065   -2.758   35.039  151.509 
# 
# Coefficients:
#               Estimate Std. Error t value Pr(>|t|)    
#   (Intercept)  149.920      2.976  50.382  < 2e-16 ***
#   age          -66.758     68.946  -0.968  0.33364    
#   sex         -304.651     69.847  -4.362 1.74e-05 ***
#   bmi          518.663     76.573   6.773 6.01e-11 ***
#   map          388.111     72.755   5.335 1.81e-07 ***
#   tc          -815.268    537.549  -1.517  0.13034    
#   ldl          387.604    439.162   0.883  0.37811    
#   hdl          162.903    269.117   0.605  0.54539    
#   tch          323.832    186.803   1.734  0.08396 .  
#   ltg          673.620    206.888   3.256  0.00125 ** 
#   glu           94.219     79.590   1.184  0.23737    
# ---
#   Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
# 
# Residual standard error: 54.05 on 321 degrees of freedom
# Multiple R-squared:  0.5213,	Adjusted R-squared:  0.5064 
# F-statistic: 34.96 on 10 and 321 DF,  p-value: < 2.2e-16


#Mean prediction error
lm.pred <- predict(model.lm,data.test)
mean((lm.pred -data.test$y)^2)
# [1] 3111.265

#SE of MSE
sd((lm.pred -data.test$y)^2)/sqrt(length((lm.pred -data.test$y)^2))
# [1] 361.0908

#########################################################################################
#Question 2 -- Apply best subset selection using BIC to select the number of predictors #
#########################################################################################

predict.regsubsets =function (object ,newdata ,id ,...){
  form=as.formula(object$call [[2]])
  mat=model.matrix(form ,newdata )
  coefi =coef(object ,id=id)
  xvars =names(coefi )
  mat[,xvars ]%*% coefi
}

regfit.full=regsubsets(y~.,data.train,nvmax =10)
reg.summary <- summary(regfit.full)
which.min(reg.summary$bic)

#Explore graph
plot(reg.summary$bic ,xlab=" Number of Variables ",ylab="BIC",
     type="l")
points(6, reg.summary$bic[6], col =" red",cex =2, pch =20)

#Get variables and coefficients
coef(regfit.full ,6)

##Coefficients
# (Intercept)         sex         bmi         map          tc         tch         ltg 
#    150.1166   -306.0420    538.8274    389.0673   -379.0379    332.6735    527.5658 

#Mean Prediction Error:
regfit.pred <- predict.regsubsets(regfit.full,data.test,id=6)
mean((regfit.pred -data.test$y)^2)
# [1] 3095.483

#SE of MSE
sd((regfit.pred -data.test$y)^2)/sqrt(length((regfit.pred -data.test$y)^2))
# [1]  369.7526

#########################################################################################
#Question 3 -- Apply best subset selection using 10-fold cross-validation to select the # 
#              number of predictors                                                     #
#########################################################################################

k=10
set.seed(1306)
folds <- sample(1:k, nrow(data.train), replace = TRUE)
cv.errors <- matrix(NA ,k,10, dimnames =list(NULL , paste (1:10) ))

for(j in 1:k){
  best.fit=regsubsets(y~.,data=data.train[folds !=j,],
                      nvmax =10)
  for(i in 1:10) {
    pred=predict(best.fit ,data.train[folds ==j,], id=i)
    cv.errors[j,i]=mean( (data.train$y[folds ==j]-pred)^2)
  }
}

mean.cv.errors =apply(cv.errors ,2, mean)
min(mean.cv.errors)
# [1] 2978.907
par(mfrow =c(1,1))
plot(mean.cv.errors ,type='b')

reg.best=regsubsets(y~.,data=data.train , nvmax =10)
coef(reg.best ,6)
#Coefficients:
# (Intercept)     sex         bmi         map          tc         tch         ltg 
# 150.1166   -306.0420    538.8274    389.0673   -379.0379    332.6735    527.5658 

#Mean Prediction Error:
regbest.pred <- predict.regsubsets(reg.best,data.test,id=6)
mean((regbest.pred -data.test$y)^2)
# [1] 3095.483

#SE of MSE
sd((regbest.pred -data.test$y)^2)/sqrt(length((regbest.pred -data.test$y)^2))
# [1] 369.7526

#########################################################################################
#Question 4 -- Ridge regression modeling using 10-fold cross-validation to select the   # 
#              largest value of λ such that the cross-validation error is within 1 std  #
#              error of the minimum                                                     # 
#########################################################################################

#This is not needed since we are using cv.out to determine the best lambda
# However, it was used to generate the plot
# grid =10^seq(10,-2, length =100)
# ridge.mod =glmnet(x.train,y.train,alpha =0, lambda =grid)
# plot(ridge.mod, xvar="lambda", label=T)

set.seed(1306)
cv.out <- cv.glmnet(x.train,y.train,alpha =0)
bestlam <- cv.out$lambda.1se
bestlam
# [1] 41.67209

#Fit model
ridge.mod =glmnet(x.train,y.train,alpha =0, lambda = bestlam)

#Mean Prediction Error
ridge.pred=predict(ridge.mod ,newx=x.test)

mean((ridge.pred-y.test)^2)
# [1] 3070.87

#SE of MSE
sd((ridge.pred -y.test)^2)/sqrt(length((ridge.pred -y.test)^2))
# [1] 350.5467

#Coefficients:
ridge.coef= coef(ridge.mod)
ridge.coef
# s0
# (Intercept)  149.99068
# age          -11.33162
# sex         -156.91053
# bmi          374.44939
# map          264.89998
# tc           -31.96990
# ldl          -66.89724
# hdl         -174.01202
# tch          123.97204
# ltg          307.68646
# glu          134.48120

#########################################################################################
#Question 5 -- Lasso model using 10-fold cross-validation to select the largest  value  # 
#              of λ such that the cross-validation error is within 1 standard error of  #
#              the minimum                                                              # 
#########################################################################################

# grid =10^seq(10,-2, length =100)
# lasso.mod =glmnet(x.train,y.train,alpha =1, lambda =grid)
# plot(lasso.mod, xvar="lambda", label=T)

set.seed(1306)
cv.out <- cv.glmnet(x.train,y.train,alpha =1)
bestlam <- cv.out$lambda.1se
bestlam
# [1] 4.791278

lasso.mod =glmnet(x.train,y.train,alpha =1, lambda = bestlam)

#Mean Prediction Error:
lasso.pred=predict(lasso.mod ,s=bestlam ,newx=x.test)
mean((lasso.pred -y.test)^2)
# [1] 2920.041

#SE of MSE
sd((lasso.pred -y.test)^2)/sqrt(length((lasso.pred -y.test)^2))
# [1] 346.2248

##Coefficients:
lasso.coef=coef(lasso.mod)
lasso.coef
# s0
# (Intercept)  149.95298
# age            .      
# sex         -119.62208
# bmi          501.56473
# map          270.92614
# tc             .      
# ldl            .      
# hdl         -180.29437
# tch            .      
# ltg          390.55001
# glu           16.58881
# submit the csv file in Canvas for evaluation based on actual test donr and damt values
  
  
  #### 3. Develop a prediction model for the DAMT variable
  ####MODEL ONE: LEAST SQUARES REGRESSION
  # Linear model with all variables.
  model.lin1 <- lm(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc + genf + wrat + 
                      +                   avhv + incm + inca + plow + npro + tgif + lgif + rgif + tdon + tlag + agif, 
                    +                 data.train.std.y)
  summary(model.lin1)
  
    # Use regsubsets() to identify the best subset of predictor variables.
    library(leaps)
  regfit.full <- regsubsets(damt ~ reg1 + reg2 + reg3 + reg4 + home + chld + hinc +
     +  wrat + genf + avhv + incm + inca + plow + npro + tgif +
     +  lgif + rgif + tdon + tlag + agif, data.train.std.y, nvmax = 20)
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
                      +                   incm + inca + plow + tgif + lgif + rgif + tdon + agif,
                    +                 data.train.std.y)
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
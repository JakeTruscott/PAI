################################################################################
#Prediction as Inference (PAI) Package - R
#Updating Sandbox Materials
#Code Developed by Ben Johnson (UF), Logan Strother (Purdue), and Jake Truscott (Purdue)
#Updated December 2023
#Contact: jtruscot@purdue.edu
################################################################################

################################################################################
#Load Packages
################################################################################
library(randomForest); library(doParallel); library(caret); library(parallel); library(rlist); library(dplyr); library(gridExtra); library(gridtext); library(grid); library(doSNOW)

################################################################################
#Develop Sandbox Data
################################################################################

{
  set.seed(1234) #Set Random Seed Generator
  seed <- 1234
  numCores <- parallel::detectCores()/2 #Generate Core Allocation (Adjust as Necessary)

  {
    numvars = 10
    train.set = 200
    test.set = 50
    obs = test.set + train.set
    sd = 1

    betas <- rnorm(numvars, sd=sd)
    exes <- matrix(rnorm(numvars*obs, sd=sd), ncol = numvars)

    noise <- rnorm(obs)
    ystar0 <- (betas %*% t(exes))[1,] + noise
    ydisc <- 1*(ystar0>0)

  } #Base Parameters
  {
    x0 <- rnorm(obs)
    ystar <- ystar0 + 2*x0
    ydisc0 <- 1*(ystar>0)
    ystar.lin <- ystar
  } #Linear Variables
  {
    x1 <- rnorm(obs, 1, 1)
    x2 <- rnorm(obs,1,1)
    intereact.beta = 10
    ystar <- ystar0 + intereact.beta*x1*x2
    ydisc1 <- 1*(ystar>0)
    ystar.inter <- ystar
  } #Interaction Term
  {
    x3 <- rnorm(obs)
    ystar <- ystar0 + 3*x3^2
    ydisc2 <- 1*(ystar>0)
    ystar.sq <- ystar
  } #Square Term
  {
    x4 <- rnorm(obs)
    x5 <- rnorm(obs)

    ystar <- ystar0 + 2*x4^3 - 6*x5^5
    ydisc3 <- 1*(ystar>0)
    ystar.poly <- ystar

    x6 <- rnorm(obs)

    ystar <- ystar0 + 2^x6
    ydisc4 <- 1*(ystar>0)
    ystar.exp <- ystar

  } #Higher-Order Polynomials
  {
    x7 <- rnorm(obs)

    ystar <- ystar0 + abs(x7)
    ydisc5 <- 1*(ystar>0)
    ystar.abs <- ystar
  } #Continuous Effect
  {
    x8 <- rnorm(obs)

    ystar <- ystar0 + sin(x8*pi)
    ydisc6 <- 1*(ystar>0)
    ystar.sin <- ystar
  } #Add Sin
  { x9 <- rnorm(obs)
    x9.1 <- x9
    x9.1[x9.1 > quantile(x9.1,.2) & x9.1 < quantile(x9.1,.4)] = quantile(x9.1,.2)
    x9.1[x9.1 > quantile(x9.1,.6) & x9.1 < quantile(x9.1,.8)] = quantile(x9.1,.6)

    ystar <- ystar0 + 4*x9.1
    ydisc7 <- 1*(ystar>0)
    ystar.mono <- ystar
  } #Weekly Monotonic Effect
  {
    ch0 <- sum(1*(ydisc != ydisc0))
    ch1 <- sum(1*(ydisc != ydisc1))
    ch2 <- sum(1*(ydisc != ydisc2))
    ch3 <- sum(1*(ydisc != ydisc3))
    ch4 <- sum(1*(ydisc != ydisc4))
    ch5 <- sum(1*(ydisc != ydisc5))
    ch6 <- sum(1*(ydisc != ydisc6))
    ch7 <- sum(1*(ydisc != ydisc7))

    changes <- c(ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7)
  } #Track How Dichotomous Observations Changed w/ Each Step
  {
    changes <- c(ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7)

    X <- as.data.frame(cbind(exes, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9))
    X.train <- X[seq(train.set),]
    X.test <- X[-seq(train.set),]
  } #Combine to DF

} #Sandbox Data

################################################################################
#Functions
################################################################################

{
  runpred <- function(mod, var, stepper, dat){
    dat[[var]] <- dat[[var]] + stepper
    pred <- predict(mod, dat)
    true <- dat$y
    onecount <- length(which(pred == '1')) / length(true)
    acc <- length(which(pred == true)) / length(true)
    return(c(onecount, acc))
  }

  runpred.bin <- function(mod, var, stepper, dat){
    dat[[var]] <- dat[[var]] + stepper
    pred <- predict(mod, dat)
    true <- dat$y
    dif <- pred - true
    return(dif)
  }

  acc.imp <- function(w, wo, w.dat, wo.dat, y){
    pred.w <- predict(w, w.dat)
    pred.wo <- predict(wo, wo.dat)
    w.acc <- sum(1 * (pred.w == y)) / length(y)
    wo.acc <- sum(1 * (pred.wo == y)) / length(y)
    imp.list <- list(w.acc = w.acc, wo.acc = wo.acc, dif = w.acc - wo.acc)
    return(imp.list)
  }

  push.bin <- function(l){
    x = l$var
    Z = l$with.test
    sdx <- sd(Z[, x])
    steps <- seq(-2 * sdx, 2 * sdx, (4 * sdx) / 100)
    tester <- lapply(steps, function(z) runpred(l$with, x, z, Z))
    t <- cbind(steps, t(list.cbind(tester)))
    return(t)
  }

  thisforest.bin <- function(y, x, p, Z = X, ntrain = train.set){
    set.seed(254)
    true.beta <- betas[x + 1]
    x = x + numvars + 2
    p = p + numvars + 2
    y = as.factor(y)

    set.seed(seed)
    registerDoParallel(numCores)
    folds <- 5
    d <- cbind(y, Z)
    d = as.data.frame(d)
    names(d)[1] = 'y'
    d$y <- as.factor(d$y)
    d.train <- d[seq(ntrain), ]
    d.test <- d[-seq(ntrain), ]
    tc <- trainControl(method = 'oob',
                       savePredictions = TRUE)
    mod.with <- train(y ~ ., method = "parRF", data = d.train,
                      trControl = tc, importance = TRUE,
                      localImp = TRUE)
    rf.basic <- randomForest(y ~ .,  data = d.train,
                             na.action = "na.omit",
                             ntree = 1000, nodesize = 4, importance = TRUE,
                             localImp = TRUE)
    without.d.train <- d.train
    without.d.train[, x] = 0
    without.d.test <- d.test
    without.d.test[, x] = 0
    mod.without <- train(y ~ ., data = without.d.train, method = "parRF",
                         trControl = tc, importance = TRUE)
    acc.ch <- acc.imp(mod.with, mod.without, d.test, without.d.test, d.test$y)
    olist <- list(with = mod.with, without = mod.without,
                  with.test = d.test, without.test = without.d.test,
                  X = X, var = p, training.data.with = d.train,
                  baseRF = rf.basic, acc.imp = acc.ch$dif)
    pusher <- push.bin(olist)
    olist$push = pusher
    modal <- max(table(d.test$y)) / sum(table(d.test$y))
    olist$outrow <- c(modal, acc.ch$w.acc, acc.ch$wo.acc)

    return(olist)
  }

  runpred.cont <- function(mod, var, stepper, dat){
    dat[[var]] <- dat[[var]] + stepper
    pred <- predict(mod, dat)
    true <- dat$y
    dif <- pred - true
    return(dif)
  }

  push.cont <- function(l){
    x = l$var
    Z = l$with.test
    sdx <- sd(Z[, x])
    steps <- seq(-2 * sdx, 2 * sdx, (4 * sdx) / 100)
    tester <- lapply(steps, function(z) runpred.cont(l$with, x, z, Z))
    t <- cbind(steps, t(list.cbind(tester)))
    return(t)
  }

  thisforest.cont <- function(y, x, p, Z = X, ntrain = train.set){
    set.seed(254)
    true.beta <- betas[x + 1]
    x = x + numvars + 2
    p = p + numvars + 2

    set.seed(seed)
    registerDoParallel(numCores)
    folds <- 5
    d <- cbind(y, Z)
    d = as.data.frame(d)
    names(d)[1] = 'y'
    d.train <- d[seq(ntrain), ]
    d.test <- d[-seq(ntrain), ]
    tc <- trainControl(method = 'oob',
                       savePredictions = TRUE)
    mod.with <- train(y ~ ., method = "ranger", data = d.train,
                      trControl = tc)

    without.d.train <- d.train
    without.d.train[, x] = 0
    without.d.test <- d.test
    without.d.test[, x] = 0
    mod.without <- train(y ~ ., data = without.d.train, method = "ranger",
                         trControl = tc)
    pred.w <- predict(mod.with, d.test)
    pred.wo <- predict(mod.without, without.d.test)

    olist <- list(with = mod.with, without = mod.without,
                  with.test = d.test, without.test = without.d.test,
                  X = X, var = p, training.data.with = d.train,
                  kiv = d.test[, x],
                  pred.w = pred.w, pred.wo = pred.wo, test.y = d.test$y)
    pusher <- push.cont(olist)
    olist$push = pusher

    return(olist)
  }

} #All Functions (Binomial & Cont)

################################################################################
#Develop Main Function -- Flexible w/ Multiple IV Allotment
#Just Random Forest
#Just Binomial
#Just Linear
################################################################################

pai_main <- function(data, #Dataframe
                     outcome = NULL, #Character or Character Vector
                     predictors = NULL, #Character or Character Vector
                     model = NULL, #Model Type ("Linear") or Vector of Models
                     ml = c(NA, NA), #Vector List of ML Model (Character) + Folds (Numeric) + Cores (Numeric)
                     seed = NULL){ #Numeric Seed

  message("-------------------------------------------------------------------")
  cat("---------------------- Beginning PAI Process ----------------------\n")
  message("-------------------------------------------------------------------")


  {
    if(is.null(outcome)){
      stop("No Dependent Variable Declared \nDeclare Dependent Variable(s) and Try Again")
    } else {
      message("    Dependent Variable = ", outcome)
    } #Print Message for DV Assignment

    if (identical(predictors, "all") || is.null(predictors)) {
      message("    NULL or 'all' Predictors Declared...\n      Compiling Predictions With All Variables Except Dependent Variable")
      predictors = names(data[!names(data) %in% outcome])
    } else {
      if (length(predictors) > 1){
        message("    Predictors = ", paste0(predictors, collapse = ", "))
      } else {
        message("    Predictors = ", predictors)
      }
    } #Print Message for IVs

    if(is.null(model)){
      model = "linear"
    } else if (model %in% c("linear", "lm")){
      model = "linear"
    } #Declare Model Type


    if(is.na(ml[1])){
      ml[1] <- "Random Forest"
    } else if (ml[1] %in% c('Random Forest', 'random forest', 'rf')) {
      ml[1] <- "Random Forest"
    } #Declare ml

    if (is.na(ml[2])) {
      ml[2] <- 2
    } else {
      ml[2] <- as.numeric(ml[2])
    } #Declare Cores

    numCores <- as.numeric(ml[2])

    if (any(is.na(ml[c(1, 2)]))) {
      message("    NULL or Default Assigned Within 1 or More Parameters in 'ml'")
      message('    ML Parameters:')
      message("          ML Model: ", ml[1])
      message("          Cores: ", ml[2])
    } else {
      message('    ML Parameters:')
      message("          ML Model: ", ml[1])
      message("          Cores: ", ml[2])
    }

    unique_dv_values <- data[[outcome]]

    if(length(unique(unique_dv_values)) == 2){
      data_type <- "Binomial"
    } else {
      data_type <- "Continuous"
    }

    if(is.null(seed)){
      seed <- 1234
    } else {
      seed <- as.numeric(seed)
    }



  } #Assigning Parameters

  dat <- data %>%
    select(any_of(c(outcome, predictors)))

  sandbox.models <- list()

  if (data_type == 'Binomial'){
    {
      {
        numvars = length(predictors)
        train.set <- round(length(dat[[outcome]])/5, 0)
        test.set <- length(dat[[outcome]]) - train.set
        obs <- length(dat[[outcome]])

        predictor_columns <- names(data[names(data) %in% predictors]) #Subset Predictor Columns
        formula_str <- paste("y ~", paste(predictor_columns, collapse = " + ")) #Create a String for a Linear Model
        lm_model <- lm(as.formula(formula_str), data = data) #Run the Linear Model
        betas <- lm_model$coefficients #Retrieve Betas
        betas <- betas[names(betas) %in% predictors]
        x0 <- lm_model$coefficients[["(Intercept)"]]

        noise <- rnorm(obs) #Create Noise
        exes <- data[names(data) %in% predictors]


        ystar0 <- (betas %*% t(exes))[1,] + noise
        ydisc <- 1*(ystar0>0)
      } #Get Set Parameters - Retrieve Betas, Y*, etc.
      {
        x0 <- rnorm(obs)
        ystar <- ystar0 + 2*x0
        ydisc0 <- 1*(ystar>0)
        ystar.lin <- ystar
      } #Linear Variables
      {
        x1 <- rnorm(obs, 1, 1)
        x2 <- rnorm(obs,1,1)
        intereact.beta = 10
        ystar <- ystar0 + intereact.beta*x1*x2
        ydisc1 <- 1*(ystar>0)
        ystar.inter <- ystar
      } #Interaction Term
      {
        x3 <- rnorm(obs)
        ystar <- ystar0 + 3*x3^2
        ydisc2 <- 1*(ystar>0)
        ystar.sq <- ystar
      } #Square Term
      {
        x4 <- rnorm(obs)
        x5 <- rnorm(obs)

        ystar <- ystar0 + 2*x4^3 - 6*x5^5
        ydisc3 <- 1*(ystar>0)
        ystar.poly <- ystar

        x6 <- rnorm(obs)

        ystar <- ystar0 + 2^x6
        ydisc4 <- 1*(ystar>0)
        ystar.exp <- ystar

      } #Higher-Order Polynomials
      {
        x7 <- rnorm(obs)

        ystar <- ystar0 + abs(x7)
        ydisc5 <- 1*(ystar>0)
        ystar.abs <- ystar
      } #Continuous Effect
      {
        x8 <- rnorm(obs)

        ystar <- ystar0 + sin(x8*pi)
        ydisc6 <- 1*(ystar>0)
        ystar.sin <- ystar
      } #Add Sin
      { x9 <- rnorm(obs)
        x9.1 <- x9
        x9.1[x9.1 > quantile(x9.1,.2) & x9.1 < quantile(x9.1,.4)] = quantile(x9.1,.2)
        x9.1[x9.1 > quantile(x9.1,.6) & x9.1 < quantile(x9.1,.8)] = quantile(x9.1,.6)

        ystar <- ystar0 + 4*x9.1
        ydisc7 <- 1*(ystar>0)
        ystar.mono <- ystar
      } #Weekly Monotonic Effect
      {
        ch0 <- sum(1*(ydisc != ydisc0))
        ch1 <- sum(1*(ydisc != ydisc1))
        ch2 <- sum(1*(ydisc != ydisc2))
        ch3 <- sum(1*(ydisc != ydisc3))
        ch4 <- sum(1*(ydisc != ydisc4))
        ch5 <- sum(1*(ydisc != ydisc5))
        ch6 <- sum(1*(ydisc != ydisc6))
        ch7 <- sum(1*(ydisc != ydisc7))

        changes <- c(ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7)
      } #Track How Dichotomous Observations Changed w/ Each Step
      {
        changes <- c(ch0, ch1, ch2, ch3, ch4, ch5, ch6, ch7)

        X <- as.data.frame(cbind(exes, x0, x1, x2, x3, x4, x5, x6, x7, x8, x9))
        X.train <- X[seq(train.set),]
        X.test <- X[-seq(train.set),]
      } #Combine to DF
      {

        runpred <- function(mod, var, stepper, dat){
          dat[[var]] <- dat[[var]] + stepper
          pred <- predict(mod, dat)
          true <- dat$y
          onecount <- length(which(pred == '1')) / length(true)
          acc <- length(which(pred == true)) / length(true)
          return(c(onecount, acc))
        }

        runpred.bin <- function(mod, var, stepper, dat){
          dat[[var]] <- dat[[var]] + stepper
          pred <- predict(mod, dat)
          true <- dat$y
          dif <- pred - true
          return(dif)
        }

        acc.imp <- function(w, wo, w.dat, wo.dat, y){
          pred.w <- predict(w, w.dat)
          pred.wo <- predict(wo, wo.dat)
          w.acc <- sum(1 * (pred.w == y)) / length(y)
          wo.acc <- sum(1 * (pred.wo == y)) / length(y)
          imp.list <- list(w.acc = w.acc, wo.acc = wo.acc, dif = w.acc - wo.acc)
          return(imp.list)
        }

        push.bin <- function(l){
          x = l$var
          Z = l$with.test
          sdx <- sd(Z[, x])
          steps <- seq(-2 * sdx, 2 * sdx, (4 * sdx) / 100)
          tester <- lapply(steps, function(z) runpred(l$with, x, z, Z))
          t <- cbind(steps, t(list.cbind(tester)))
          return(t)
        }

        thisforest.bin <- function(y, x, p, Z = X, ntrain = train.set) {

          set.seed(seed)
          true.beta <- betas[x + 1]
          x = x + numvars + 2
          p = p + numvars + 2
          y = as.factor(y)

          cl <- makePSOCKcluster(numCores)
          registerDoParallel(cl)
          folds <- as.numeric(ml[2])
          d <- cbind(y, Z)
          d = as.data.frame(d)
          names(d)[1] = 'y'

          d$y <- as.factor(d$y)
          d.train <- d[seq(ntrain), ]
          d.test <- d[-seq(ntrain), ]
          tc <- trainControl(method = 'oob',
                             savePredictions = TRUE)

          mod.with <- train(y ~ ., method = "parRF", data = d.train,
                            trControl = tc, importance = TRUE,
                            localImp = TRUE, verbose = FALSE)
          rf.basic <- randomForest(y ~ ., data = d.train,
                                   na.action = "na.omit",
                                   ntree = 1000, nodesize = 4, importance = TRUE,
                                   localImp = TRUE)
          without.d.train <- d.train
          without.d.train[, x] = 0
          without.d.test <- d.test
          without.d.test[, x] = 0
          mod.without <- train(y ~ ., data = without.d.train, method = "parRF",
                               trControl = tc, importance = TRUE, verbose = F)
          acc.ch <- acc.imp(mod.with, mod.without, d.test, without.d.test, d.test$y)
          olist <- list(with = mod.with, without = mod.without,
                        with.test = d.test, without.test = without.d.test,
                        X = X, var = p, training.data.with = d.train,
                        baseRF = rf.basic, acc.imp = acc.ch$dif)
          pusher <- push.bin(olist)

          olist$push = pusher
          modal <- max(table(d.test$y)) / sum(table(d.test$y))
          olist$outrow <- c(modal, acc.ch$w.acc, acc.ch$wo.acc)

          return(olist)
        }



      } #Binary Data Functions
      {
        sandbox.models.bin <- list()
        sandbox.models$bin$linear <- thisforest.bin(ydisc0, 0, 0)
        sandbox.models$bin$interact <- thisforest.bin(ydisc1, c(1,2), 1)
        sandbox.models$bin$square <- thisforest.bin(ydisc2, 3, 3)
        sandbox.models$bin$poly <- thisforest.bin(ydisc3, c(4,5), 5)
        sandbox.models$bin$polySmall <- thisforest.bin(ydisc3, c(4,5), 4)
        sandbox.models$bin$exp <- thisforest.bin(ydisc4, 6, 6)
        sandbox.models$bin$abs <- thisforest.bin(ydisc5, 7, 7)
        sandbox.models$bin$sin <- thisforest.bin(ydisc6, 8, 8)
        sandbox.models$bin$mono <- thisforest.bin(ydisc7, 9, 9)
      } #Allocate Outputs to List
    } #Binomial

  } else {
    {
      {
        numvars = length(predictors)
        train.set <- round(length(dat[[outcome]])/5, 0)
        test.set <- length(dat[[outcome]]) - train.set
        obs <- length(dat[[outcome]])

        predictor_columns <- names(data[names(data) %in% predictors]) #Subset Predictor Columns
        formula_str <- paste("y ~", paste(predictor_columns, collapse = " + ")) #Create a String for a Linear Model
        lm_model <- lm(as.formula(formula_str), data = data) #Run the Linear Model
        betas <- lm_model$coefficients #Retrieve Betas
        betas <- betas[names(betas) %in% predictors]
        x0 <- lm_model$coefficients[["(Intercept)"]]

        noise <- rnorm(obs) #Create Noise
        exes <- data[names(data) %in% predictors]


        ystar0 <- (betas %*% t(exes))[1,] + noise
        ydisc <- 1*(ystar0>0)
      } #Get Set Parameters - Retrieve Betas, Y*, etc.
      {
        #x0 <- rnorm(obs)
        ystar <- ystar0 + 2*x0
        ydisc0 <- ydisc
        ystar.lin <- ystar
      } #Linear Variables
      {
        X <- dat
        X.train <- X[seq(train.set),]
        X.test <- X[-seq(train.set),]
      } #Combine to DF
      {
        runpred.cont <- function(mod, var, stepper, dat){
          dat[[var]] <- dat[[var]] + stepper
          pred <- predict(mod, dat)
          true <- dat$y
          dif <- pred - true
          return(dif)
        }
        acc.imp <- function(w, wo, w.dat, wo.dat, y){
          pred.w <- predict(w, w.dat)
          pred.wo <- predict(wo, wo.dat)
          w.acc <- sum(1*(pred.w==y)) / length(y)
          wo.acc <- sum(1*(pred.wo==y)) / length(y)
          imp.list <- list(w.acc=w.acc, wo.acc=wo.acc, dif=w.acc-wo.acc)
          return(imp.list)
        }
        push.cont <- function(l){
          t <- list()

          for (v in 1:length(l$var)){
            x = l$var[v]
            Z = l$with.test
            sdx <- sd(Z[,x])
            steps <- seq(-2*sdx, 2*sdx, (4*sdx)/100)
            tester <- lapply(steps, function(z) runpred.cont(l$with, x, z, Z ))
            temp <- cbind(steps, t(list.cbind(tester)))
            t[[x]] <- temp
          }

          return(t)
        }
        thisforest.cont <- function(y, predictors, Z = X, ntrain = train.set) {
          set.seed(seed)

          registerDoParallel(numCores)

          Z <- Z[names(Z) %in% predictors]

          d <- cbind(y, Z)
          d = as.data.frame(d)

          tc <- trainControl(method = 'oob', savePredictions = TRUE)

          without.d.train <- d[seq(ntrain), ]

          without.d.test <- d[-seq(ntrain), ]

          capture_output_mod.with <- capture.output({
            mod.with <- train(y ~ ., method = "ranger", data = d[seq(ntrain),], trControl = tc)
          })

          capture_output_mod.without <- capture.output({
            mod.without <- train(y ~ ., data = without.d.train, method = "ranger", trControl = tc)
          })


          pred.w <- predict(mod.with, d[-seq(ntrain), ])
          pred.wo <- predict(mod.without, without.d.test)

          olist <- list(
            with = mod.with,
            without = mod.without,
            with.test = d[-seq(ntrain), ],
            without.test = without.d.test,
            X = X,
            var = predictors,
            training.data.with = d[seq(ntrain), ],
            kiv = d[-seq(ntrain), predictors, drop = FALSE],
            pred.w = pred.w,
            pred.wo = pred.wo,
            test.y = d[-seq(ntrain), ]$y
          )

          pusher <- push.cont(olist)
          olist$push = pusher

          return(olist)
        }


      } #Continuous Data Functions
      {
        sandbox.models$cont <- list()
        sandbox.models$cont$linear <- thisforest.cont(y = ystar.lin, predictors = predictors, )

      } #Allocate Outputs to List
    } #Continuous
  } #PAI Sandbox Models


  message("-------------------------------------------------------------------")
  cat("-------------------------- PAI  Complete --------------------------\n")
  message("-------------------------------------------------------------------")

  return(sandbox.models)


}



################################################################################
#Function Parameters
#     data = Data Frame
#     outcome = DV: Character For Column in DF
#     predictors = IVs: Character (or Comma-Separated Vector) for Column(s) in DF
#         - Can Also Return 'all' or *empty*/*NULL* -- If 'all' or 'NULL', Will Use Every Column Other than DV
#     model = Character or Vectors of Model Type (e.g., "Linear" or "LM"; "Exp", etc.)
#     ml = Machine Learning Model (e.g., 'Random Forest' or 'rf')
#     seed = (Optional) random seed generator (Character or Numeric -- Converts to numeric in Function)


################################################################################


################################################################################
#Test w/ Sample Data
################################################################################

set.seed(1234)
test_data <- data.frame(y = sample(c(0:50), 100, replace = TRUE),
                   var1 = rnorm(100, mean = 0, sd = 1),
                   var2 = rnorm(100, mean = 1, sd = 1),
                   var3 = rnorm(100, mean = 3, sd = 2),
                   var4 = rnorm(100, mean = 0, sd = 1))

data = test_data
outcome = "y"
predictors = c("var1", "var2", "var3")
model = NULL
ml = c("Random Forest", 5, 8)
seed = 1234

test <- pai_main(data = test_data,
                 outcome = "y",
                 predictors = c("var1", "var2", "var3"),
                 model = NULL,
                 ml = c("Random Forest", 8),
                 seed = 1234)



################################################################################
#Plot Sample Output
################################################################################

pai_plot_BASE_flexible <- function(data,
                                   data_type = NULL,
                                   model_type = NULL,
                                   custom_aes = NULL) {

  {
    if (is.null(data_type)) {
      # Default to Binomial
      data <- data$bin
    } else if (data_type == "Binomial") {
      # Assign Binomial
      data <- data$bin
    } else if (data_type == "Continuous") {
      # Assign Continuous
      data <- data$cont
    } else if (data_type == "Both") {
      stop('Need to implement a function for "Both"')
      # You might want to implement a function for "Both" here
    }
  } #Subset by Data Type
  {
    if (is.null(model_type)){
      data <- data$linear # Default to Linear
    } else if (grepl('Linear', model_type, ignore.case = TRUE)){
      data <- data$linear # Assign Linear
    } else if (grepl('Interact', model_type, ignore.case = TRUE)){
      data <- data$interact # Assign Interaction
    } else if (grepl('Square', model_type, ignore.case = TRUE)){
      data <- data$square # Assign Square
    } else if (grepl('polysmall', model_type, ignore.case = TRUE)){
      data <- data$polySmall # Assign Polynomial
    } else if (grepl('poly', model_type, ignore.case = TRUE)){
      data <- data$poly # Assign Poly(Small)
    } else if (grepl('exp', model_type, ignore.case = TRUE)){
      data <- data$exp # Assign Exponential
    } else if (grepl('abs', model_type, ignore.case = TRUE)){
      data <- data$abs # Assign Abs
    } else if (grepl('sin', model_type, ignore.case = TRUE)){
      data <- data$sin # Assign Sin
    } else if (grepl('mono', model_type, ignore.case = TRUE)){
      data <- data$mono # Assign Monotonic
    }

  } #Subset by Model Type

  data <- data.frame(data$push)

  {
    plot_generator <- function(data, custom_aes){

      if (is.null(model_type)){
        model_type <- "Linear"
      }
      if (is.null(custom_aes)){
        custom_aes <- FALSE
      }

      if(!custom_aes == TRUE){
        default_plot <- data %>%
          ggplot(aes(x = data[,1], y = data[,2])) +
          geom_line(aes(x = data[,1], y = data[,2]), colour = 'deepskyblue3', size = 1) +
          geom_line(aes(x = data[,1], y = data[,3]), colour = 'coral3', size = 1) +
          theme_minimal() +
          labs(
            x = '\nStepper\n',
            y = 'Outcome\n',
            title = paste0(model_type, ' Model'),
            subtitle = "Steppers & Model Accuracy",
            caption = 'Note: Lines Should be Parallel & (Weakly) Monotonic')  +
          theme(
            axis.title = element_text(size = 15),
            axis.text = element_text(size = 15),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "bottom",
            strip.text = element_text(size = 14, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold"),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 10, hjust = 0))

        return(default_plot)

      } else {
        custom_plot <- data %>%
          ggplot(aes(x = data[,1], y = data[,2])) +
          theme_minimal()
        warning('Returned Empty Plot - Add ggplot2() Aesthetics Using "+"')
        return(custom_plot)
      }
    }  #Plot Generator
  } #Plot Generator (Function)

  plot <- plot_generator(data = data,
                         custom_aes= custom_aes)

  return(plot)

}


pai_plot_BASE_flexible(test,
                       data_type = 'Continuous',
                       model_type = "Linear")

pai_plot_BASE_flexible_multiple <- function(data,
                                            data_types = NULL,
                                            model_types = NULL){

  data_types = ifelse(grepl('Binomial', data_types, ignore.case = T), 'Binomial', 'Continuous')

  figures <- list()

  for (d in data_types){

    model_temp_figures <- list()

    for (m in model_types){

      {

        temp_figure <- pai_plot_BASE_flexible(data = data,
                                              data_type = d,
                                              model_type = m,
                                              custom_aes = F) +
          labs(y = "",
               x = "",
               caption = " ",
               title = paste0(stringr::str_to_title(m), " Model (", d, ")")) +
          theme(plot.title = element_text(size = 15, face = "bold"),
                plot.subtitle = element_text(size = 12),
                axis.text = element_text(size = 10))


      } #Compile Temp Figure



      model_temp_figures[[m]] <- temp_figure

    }

    figures[[d]] <- model_temp_figures

  }

  all_figures <- unlist(figures, recursive = FALSE)

  yleft <- textGrob("\nOutcome or Prediction", rot = 90, gp = gpar(fontsize = 15))
  bottom <- textGrob("Steps\n",  gp = gpar(fontsize = 15))


  uni <-  grid.arrange(grobs = all_figures, left = yleft, bottom = bottom)

  return(uni)

} #Code for Producing Multiple Figures w/ Flexible Selection

pai_plot_BASE_flexible_multiple(data = test,
                                data_type = 'continuous',
                                model_types = c('linear', 'exp', 'sin', 'mono'))


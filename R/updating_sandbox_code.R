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
                     ml = c(NA, NA, NA), #Vector List of ML Model (Character) + Folds (Numeric) + Cores (Numeric)
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

    if(is.na(ml[2])){
      ml[2] <- 5
    } else {
      ml[2] <- as.numeric(ml[2])
    } #Declare Folds

    if (is.na(ml[3])) {
      ml[3] <- 2
    } else {
      ml[3] <- as.numeric(ml[3])
    } #Declare Cores

    numCores <- as.numeric(ml[3])

    if (any(is.na(ml[c(1, 2, 3)]))) {
      message("    NULL or Default Assigned Within 1 or More Parameters in 'ml'")
      message('    ML Parameters:')
      message("          ML Model: ", ml[1])
      message("          Folds: ", ml[2])
      message("          Cores: ", ml[3])
    } else {
      message('    ML Parameters:')
      message("          ML Model: ", ml[1])
      message("          Folds: ", ml[2])
      message("          Cores: ", ml[3])
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


  if (model == "linear") {
    if (data_type == "Binomial"){

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
        ystar <- ystar0 + 2*x0
        ydisc0 <- 1*(ystar>0)
        ystar.lin <- ystar
      } #Linear Variables
      {
        ch0 <- sum(1*(ydisc != ydisc0))
        changes <- c(ch0)

      } #Track How Dichotomous Observations Changed w/ Each Step
      {
        X <- as.data.frame(cbind(exes, x0))
        X.train <- X[seq(train.set),]
        X.test <- X[-seq(train.set),]
      } #Combine to DF - Get Training & Test Data
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
            rf.basic <- randomForest(y ~ .,  data = d.train,
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

      sandbox.models$linear <- suppressMessages(thisforest.bin(ydisc0, 0, 0))

    }
  }

  message("-------------------------------------------------------------------")
  cat("-------------------------- PAI  Complete --------------------------\n")
  message("-------------------------------------------------------------------")

  return(sandbox.models)
  stopCluster(cl)


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

test_data <- data.frame(y = sample(c(0, 1), 100, replace = TRUE),
                   var1 = runif(100, 0, 100),
                   var2 = runif(100, 0, 150),
                   var3 = runif(100, 0, 120),
                   var4 = runif(100, 100, 230))

test <- pai_main(data = test_data,
                 outcome = "y",
                 predictors = c("var1", "var2", "var3"),
                 model = NULL,
                 ml = c("Random Forest", 1, 8),
                 seed = 1234)






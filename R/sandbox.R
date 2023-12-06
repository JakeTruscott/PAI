################################################################################
#Prediction as Inference (PAI) Package - R
#Code Developed by Ben Johnson (UF), Logan Strother (Purdue), and Jake Truscott (Purdue)
#Updated December 2023
#Contact: jtruscot@purdue.edu
################################################################################

################################################################################
#Load Packages
################################################################################
library(randomForest); library(doParallel); library(caret); library(parallel); library(rlist); library(dplyr); library(gridExtra); library(gridtext); library(grid)


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

runpred <- function(mod, var, stepper, dat){
  dat[[var]] <- dat[[var]] + stepper
  pred <- predict(mod, dat)
  true <- dat$y
  onecount <- length(which(pred=='1'))/length(true)
  acc <- length(which(pred==true))/length(true)
  olist <- list(pred=pred, true=true, ones=onecount, acc=acc)
  #return(olist)
  return(c(onecount, acc))
}

runpred.cont <- function(mod, var, stepper, dat){
  dat[[var]] <- dat[[var]] + stepper
  pred <- predict(mod, dat)
  true <- dat$y
  dif <- pred - true
  return(dif)
  #olist <- list(pred=pred, true=true, dif=dif)
  #return(olist)
  #return(c(onecount, acc))
}

acc.imp <- function(w, wo, w.dat, wo.dat, y){
  pred.w <- predict(w, w.dat)
  pred.wo <- predict(wo, wo.dat)
  w.acc <- sum(1*(pred.w==y)) / length(y)
  wo.acc <- sum(1*(pred.wo==y)) / length(y)
  imp.list <- list(w.acc=w.acc, wo.acc=wo.acc, dif=w.acc-wo.acc)
  return(imp.list)
}

push.bin <- function(l){
  x = l$var #x+numvars+11
  #y=as.factor(y)
  Z = l$with.test
  sdx <- sd(Z[,x])
  steps <- seq(-2*sdx, 2*sdx, (4*sdx)/100)
  tester <- lapply(steps, function(z) runpred(l$with, x, z, Z ))
  t <- cbind(steps, t(list.cbind(tester)))
  return(t)
}

push.cont <- function(l){
  x = l$var #x+numvars+11
  #y=as.factor(y)
  Z = l$with.test
  sdx <- sd(Z[,x])
  steps <- seq(-2*sdx, 2*sdx, (4*sdx)/100)
  tester <- lapply(steps, function(z) runpred.cont(l$with, x, z, Z ))
  t <- cbind(steps, t(list.cbind(tester)))
  return(t)
}

thisforest.bin <- function(y, x, p, Z=X, ntrain=train.set){
  set.seed(254)
  true.beta <- betas[x+1]
  x = x+numvars+2
  p = p+numvars+2
  y=as.factor(y)

  set.seed(seed)
  registerDoParallel(numCores)
  folds <- 5
  d <- cbind(y,Z)
  d = as.data.frame(d)
  names(d)[1]='y'
  d$y <- as.factor(d$y)
  d.train <- d[seq(ntrain),]
  d.test <- d[-seq(ntrain),]
  tc <- trainControl(method = 'oob',
                     #number = folds,
                     savePredictions = TRUE)
  mod.with <- train(y ~ ., method = "parRF", data=d.train,
                    trControl = tc, importance=TRUE,
                    localImp=TRUE)
  rf.basic <- randomForest(y ~ .,  data=d.train,
                           na.action="na.omit",
                           ntree=1000,nodesize=4,importance=TRUE,
                           localImp=TRUE)
  #without.Z <- Z
  #without.Z[,x]=0
  #without.d <- cbind(y, without.Z)
  #without.d <- as.data.frame(without.d)
  #names(without.d)[1]='y'
  #without.d$y <- as.factor(without.d$y)
  #without.d.train <- without.d[seq(ntrain),]
  #without.d.test <- without.d[-seq(ntrain),]
  without.d.train <- d.train
  without.d.train[,x] = 0
  without.d.test <- d.test
  without.d.test[,x] = 0
  mod.without <- train(y ~ ., data=without.d.train, method = "parRF",
                       trControl = tc, importance=TRUE)
  acc.ch <- acc.imp(mod.with, mod.without, d.test, without.d.test, d.test$y)
  olist <- list(with=mod.with, without=mod.without,
                with.test=d.test, without.test=without.d.test,
                X=X, var=p, training.data.with = d.train,
                baseRF = rf.basic, acc.imp=acc.ch$dif)
  pusher <- push.bin(olist)
  olist$push = pusher
  modal <- max(table(d.test$y))/sum(table(d.test$y))
  olist$outrow <- c(modal, acc.ch$w.acc, acc.ch$wo.acc)

  return(olist)
}

thisforest.cont <- function(y, x, p, Z=X, ntrain=train.set){
  set.seed(254)
  true.beta <- betas[x+1]
  x = x+numvars+2
  p = p+numvars+2
  #y=as.factor(y)

  set.seed(seed)
  registerDoParallel(numCores)
  folds <- 5
  d <- cbind(y,Z)
  d = as.data.frame(d)
  names(d)[1]='y'
  #d$y <- as.factor(d$y)
  d.train <- d[seq(ntrain),]
  d.test <- d[-seq(ntrain),]
  tc <- trainControl(method = 'oob',
                     #number = folds,
                     savePredictions = TRUE)
  mod.with <- train(y ~ ., method = "ranger", data=d.train,
                    trControl = tc)
  #importance=TRUE,
  #localImp=TRUE)
  #rf.basic <- randomForest(y ~ .,  data=d.train,
  #                         na.action="na.omit",
  #                         ntree=1000,nodesize=4,importance=TRUE,
  #                         localImp=TRUE)

  without.d.train <- d.train
  without.d.train[,x] = 0
  without.d.test <- d.test
  without.d.test[,x] = 0
  mod.without <- train(y ~ ., data=without.d.train, method = "ranger",
                       trControl = tc)
  #acc.ch <- acc.imp(mod.with, mod.without, d.test, without.d.test, d.test$y)
  pred.w <- predict(mod.with, d.test)
  pred.wo <- predict(mod.without, without.d.test)

  olist <- list(with=mod.with, without=mod.without,
                with.test=d.test, without.test=without.d.test,
                X=X, var=p, training.data.with = d.train,
                #baseRF = rf.basic,
                kiv = d.test[,x],
                pred.w=pred.w, pred.wo=pred.wo, test.y=d.test$y)
  pusher <- push.cont(olist)
  olist$push = pusher

  return(olist)
}

makesandbox <- function(){
  sandbox.models <- list()
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

  sandbox.models$cont <- list()
  sandbox.models$cont$linear <- thisforest.cont(ystar.lin, 0, 0)
  sandbox.models$cont$interact <- thisforest.cont(ystar.inter, c(1,2), 1)
  sandbox.models$cont$square <- thisforest.cont(ystar.sq, 3, 3)
  sandbox.models$cont$poly <- thisforest.cont(ystar.poly, c(4,5), 5)
  sandbox.models$cont$polySmall <- thisforest.cont(ystar.poly, c(4,5), 4)
  sandbox.models$cont$exp <- thisforest.cont(ystar.exp, 6, 6)
  sandbox.models$cont$abs<- thisforest.cont(ystar.abs, 7, 7)
  sandbox.models$cont$sin<- thisforest.cont(ystar.sin, 8, 8)
  sandbox.models$cont$mono<- thisforest.cont(ystar.mono, 9, 9)

  return(sandbox.models)
}


"
#### What's Going on Under the Hood ####

#runpred#
- Modifies Variable (var) by step
- Runs Predictions using (mod) -- Random Forest
- Compares Predictions to True Values (dat$y)
- Calculates Proportion of Predicted Values (=1) to Overall Accuracy ('acc')
- Returns List Containing Predictions, True Values, Proportions of (=1), and Accuracy

#runpred.cont
- Does the Same As Above but with Continuous Variable

#acc.imp
- Compares Accuracy Between Two Models (w and wo) on Different Datasets
- Returns List of Accuracy for Both

#push.bin & push.cont
- Generate a series of predictions by incrementing a specified variable and storing the results.
- Used for analyzing the impact of modifying a variable on the model's predictions.

#thisforest.bin & thisforest.cont
- Builds and evaluates a binary random forest model
- Compares the model with and without modifying a specific variable
- Returns a list containing the models, test datasets, variable information, and accuracy information


#### Workflow ####

## Binomial Example ##

1) Data Prep - Include both Y (Response) and X'n (Predictors)
2) Model Training - Uses randomForest function to build binary random forest (mod.with) on training data (d.train).
3) Impact Assessment:
      - A) Create New Dataset (without.d.train) By Setting Values of Specific Variable to 0
      - B) Run Binary Random Forest (mod.without) Using Modified Data (without.d.train)
4) Model Evaluation - Using acc.imp to compare accuracy of model with and without modified variable on test data (d.test & without.d.test)
5) Impact Analysis:
      - A) The push.bin function is used to analyze the impact of modifying the variable by pushing it through a range of values and observing the changes in model predictions
6) Result Storage:
      - A) The results, including models (mod.with, mod.without), test datasets (d.test, without.d.test), variable information, and accuracy information, are stored in a list (olist)


"


################################################################################
#Initialize Function w/ Sandbox Data
################################################################################
sandbox.models <- makesandbox()
closeAllConnections()

################################################################################
#Summary of What is in Each List within 'sandbox.models'
################################################################################

### sandbox.models$bin (Binary Outcomes) ###
### sandbox.models$cont (Continuous Outcomes) ###

## Lists Nested ##

# $linear = Linear Predictor (thisforest.bin w/ ydisc0)
# $interact = Interaction (thisforest.bin with ydisc1)
# $square = Squared Term for Single Predictor (thisforest.bin with ydisc2)
# $poly = Polynomial Terms for 2 Variables (thisforest.bin with ydisc3)
# $polySmall = Poly Terms for 2 Variables w/ Smaller Degree (thisforest.bin with ydisc3)
# $exp = Exponential Term for Single Variable (thisforest.bin with ydisc4)
# $abs = Absolute Term for Single Variable (thisforest.bin with ydisc5)
# $sin = Sine Term for Single Variable (thisforest.bin with ydisc6)
# $mono = Monotonic Transformation for Single Variable (thisforest.bin with ydisc7)


################################################################################
# Functiont to Return Figure(s)
# Allows Selection of Data Type(s) - Linear or Not
# Allows Selection of Model Type (Binomial or Continuous)
################################################################################


pai_plot_BASE_binomial <- function(data,
                     custom_aes = FALSE){

  data <- data.frame(data$bin$linear$push)


  if(custom_aes == FALSE){
    default_plot <- data %>%
      ggplot(aes(x = data[,1], y = data[,2])) +
      geom_line(aes(x = data[,1], y = data[,2]), colour = 'deepskyblue3', size = 1.5) +
      geom_line(aes(x = data[,1], y = data[,2]), colour = 'coral3') +
      theme_minimal() +
      labs(
        x = '\nStepper\n',
        y = 'Prediction\n',
        title = 'Linear Model',
        subtitle = "Stepper Values & Accuracy",
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


} #Base Function (Binomial)

pai_plot_BASE_binomial(data = sandbox.models,
                       custom_aes = T) +
  geom_line(aes(x = data[,1], y = data[,2])) #Base Plot (Binomial)  w/ Custom AES

pai_plot_BASE_binomial(data = sandbox.models,
                       custom_aes = F) #Base Plot (Binomial)  w/ Default AES



################
pai_plot_BASE_continuous <- function(data,
                                   custom_aes = FALSE){

  data <- data.frame(data$cont$linear$push)


  if(custom_aes == FALSE){
    default_plot <- data %>%
      ggplot(aes(x = data[,1], y = data[,2])) +
      geom_line(aes(x = data[,1], y = data[,2]), colour = 'deepskyblue3', size = 1.5) +
      geom_line(aes(x = data[,1], y = data[,2]), colour = 'coral3') +
      theme_minimal() +
      labs(
        x = '\nStepper\n',
        y = 'Prediction\n',
        title = 'Linear Model',
        subtitle = "Stepper Values & Accuracy",
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


} #Base Function (Continuous)

pai_plot_BASE_continuous(data = sandbox.models,
                       custom_aes = F) #Base Plot (Continuous) w/ Default AES




################
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
      data <- data$linear #Default to Linear
    } else if (model_type == "Linear"){
      data <- data$linear #Assign Linear
    } else if (model_type == "Interact"){
      data <- data$interact #Assign Interaction
    } else if (model_type == "Square"){
      data <- data$square #Assign Square
    } else if (model_type == "Poly"){
      data <- data$poly #Assign Polynomial
    } else if (model_type == "PolySmall"){
      data <- data$polySmall #Assign Poly(Small)
    } else if (model_type == "Exp"){
      data <- data$exp #Assign Exponential
    } else if (model_type == "Abs"){
      data <- data$abs #Assign Abs
    } else if (model_type == "Sin"){
      data <- data$sin #Assign Sin
    } else if (model_type == "Mono"){
      data <- data$mono #Assign Monotonic
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

} #Flexible Base Code For Producing Plots w/ Selection

pai_plot_BASE_flexible(data = sandbox.models,
                       data_type = "Binomial",
                       model_type = "Linear") #Implementation of Base Code for Producing Plots w/ Selection



################
pai_plot_BASE_flexible_multiple <- function(data,
                                            data_types = NULL,
                                            model_types = NULL){

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
               title = paste0(m, " Model (", d, ")")) +
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

  cols <- length(data_types)

  uni <-  grid.arrange(grobs = all_figures, left = yleft, bottom = bottom, ncol = cols)

  return(uni)

} #Code for Producing Multiple Figures w/ Flexible Selection

pai_plot_BASE_flexible_multiple(data = sandbox.models,
                                data_types = c("Binomial", "Continuous"),
                                model_types = c("Linear", "Exp", "Mono")) #Testing w/ Multiple Data & Model Types





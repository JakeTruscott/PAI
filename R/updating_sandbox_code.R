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
library(randomForest); library(doParallel); library(caret); library(parallel); library(rlist); library(dplyr); library(gridExtra); library(gridtext); library(grid); library(doSNOW); library(patchwork)

################################################################################
#Develop Main Function -- Flexible w/ Multiple IV Allotment
################################################################################

pai_main <- function(data, #Dataframe
                     outcome = NULL, #Character or Character Vector (Required)
                     predictors = NULL, #Character or Character Vector (Optional - Default to 'all' Except DV)
                     interactions = NULL, #Character Vector (Optional - Default to Null -- Indicate as Single Character with term(s) separated by '*' -- Ex: 'var1*var2')
                     ml = c(NA, NA, NA), #Vector List of ML Model (Character) + Cores (Numeric) (Optional - Default to RF & 2) + # of Placebo Iterations
                     seed = NULL){ #Numeric Seed (Optional - Default to 1234)

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

    if (is.null(interactions)){
      message("    Interactions = None")
    } else {

      print_interactions <- c()
      for (interaction in 1:length(interactions)){
        temp_interaction <- paste0("[", interaction, "] ", interactions[interaction])
        print_interactions[interaction] <- temp_interaction
      }


      message("    Interactions = ", paste(print_interactions, collapse = '\n                   '))

    }


    if(is.na(ml[1])){
      ml[1] <- 'rf'
    } #Declare ml

    if (is.na(ml[2])) {
      ml[2] <- 2
    } else {
      ml[2] <- as.numeric(ml[2])
    } #Declare Cores

    numCores <- as.numeric(ml[2])

    if(is.na(ml[3])){
      ml[3] <- 5
    } else {
      ml[3] <- as.numeric(ml[3])
    } #Declare Placebo Iterations


    if (any(is.na(ml[c(1, 2, 3)]))) {
      message("    NULL or Default Assigned Within 1 or More Parameters in 'ml'")
      message('    ML Parameters:')
      message("          ML Model: ", ml[1])
      message("          Cores: ", ml[2])
      message("          Placebo Shuffling Repetitions: ", ml[3])

    } else {
      message('    ML Parameters:')
      message("          ML Model: ", ml[1])
      message("          Cores: ", ml[2])
      message("          Placebo Shuffling Repetitions: ", ml[3])
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

  ml_model <- ml[1]
  placebo_iterations <- as.numeric(ml[3])
  train.set <- round(length(dat[[outcome]])/5, 0)
  outcome_var <- dat[outcome]


  if (data_type == 'Continuous'){
    full_vars <- ifelse(is.null(interactions), paste0("y ~ ", paste(predictors, collapse = " + ")) , paste0("y ~ ", paste(predictors, collapse = " + "), " + ", paste(interactions, collapse = " + ")))
  } else {
    full_vars <- ifelse(is.null(interactions), paste0("as.factor(y) ~ ", paste(predictors, collapse = " + ")) , paste0("as.factor(y) ~ ", paste(predictors, collapse = " + "), " + ", paste(interactions, collapse = " + ")))
  }



  {

    runpred.bin <- function(mod, var, stepper, dat){
      dat[[var]] <- dat[[var]] + stepper
      pred <- predict(mod, dat)
      true <- as.factor(dat$y)  # Ensure true is a factor
      dif <- as.numeric(pred) - as.numeric(true)  # Convert factor to numeric for calculation
      return(dif)
    }
    runpred.cont <- function(mod, var, stepper, dat){
      dat[[var]] <- dat[[var]] + stepper
      pred <- predict(mod, dat)
      true <- dat$y
      dif <- pred - true
      return(dif)
    }

    acc.imp.bin <- function(w, wo, w.dat, wo.dat, y){
      pred.w <- predict(w, w.dat)
      pred.wo <- predict(wo, wo.dat)
      w.acc <- sum(1 * (pred.w == y)) / length(y)
      wo.acc <- sum(1 * (pred.wo == y)) / length(y)
      imp.list <- list(w.acc = w.acc, wo.acc = wo.acc, dif = w.acc - wo.acc)
      return(imp.list)
    }
    acc.imp.cont <- function(w, wo, w.dat, wo.dat, y){
      pred.w <- predict(w, w.dat)
      pred.wo <- predict(wo, wo.dat)
      w.acc <- sum(1*(pred.w==y)) / length(y)
      wo.acc <- sum(1*(pred.wo==y)) / length(y)
      imp.list <- list(w.acc=w.acc, wo.acc=wo.acc, dif=w.acc-wo.acc)
      return(imp.list)
    }

    push.bin <- function(l){
      t <- list()

      for (v in 1:length(l$var)){
        x = l$var[v]
        Z = l$with.test
        sdx <- sd(Z[,x])
        steps <- seq(-2*sdx, 2*sdx, (4*sdx)/100)
        tester <- lapply(steps, function(z) runpred.bin(l$with, x, z, Z ))
        temp <- cbind(steps, t(list.cbind(tester)))
        t[[x]] <- temp
      }

      return(t)
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

    placebo_shuffle.bin <- function(w, d.train, d.test){
      message("    Beginning Placebo Protocol...")

      tc <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, savePredictions = TRUE)

      replications <- placebo_iterations
      placebos <- data.frame()

      variables <- setdiff(names(d.test), "y")

      for (rep_count in 1:replications) {
        for (variable in variables) {
          capture_output_original_accuracy <- capture.output({ suppressWarnings(original_accuracy <- train(as.formula(full_vars),
                                                                                                           data = d.test,
                                                                                                           method = as.character(ml[1]),
                                                                                                           trControl = tc)) })
          original_accuracy <- original_accuracy$results$Accuracy

          shuffle_data <- d.test
          shuffle_data[[variable]] <- sample(shuffle_data[[variable]])

          capture_output_shuffled_accuracy <- capture.output({ suppressWarnings( shuffled_accuracy <- train(as.formula(full_vars),
                                                                                                            data = shuffle_data,
                                                                                                            method = as.character(ml[1]),
                                                                                                            trControl = tc) )})
          shuffled_accuracy <- shuffled_accuracy$results$Accuracy

          accuracy_change <- original_accuracy - shuffled_accuracy

          placebo_temp <- data.frame(rep_count = rep_count, variable = variable, accuracy_change = accuracy_change)
          placebos <- bind_rows(placebos, placebo_temp)
        }

        if (rep_count %% 5){
          cat(paste0('           Completed Iteration ', rep_count, '\n'))
        }

      }

      return(placebos)

    }
    placebo_shuffle.cont <- function(w, d.train, d.test){
      message("    Beginning Placebo Protocol...")

      tc <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, savePredictions = TRUE)

      replications <- placebo_iterations
      placebos <- data.frame()

      variables <- setdiff(names(d.test), "y")

      for (rep_count in 1:replications) {
        for (variable in variables) {
          capture_output_original_accuracy <- capture.output({ suppressWarnings(original_accuracy <- train(as.formula(full_vars),
                                                                                                           data = d.test,
                                                                                                           method = as.character(ml[1]),
                                                                                                           trControl = tc)) })
          original_accuracy <- original_accuracy$results$RMSE

          shuffle_data <- d.test
          shuffle_data[[variable]] <- sample(shuffle_data[[variable]])

          capture_output_shuffled_accuracy <- capture.output({ suppressWarnings( shuffled_accuracy <- train(as.formula(full_vars),
                                                                                                            data = shuffle_data,
                                                                                                            method = as.character(ml[1]),
                                                                                                            trControl = tc) )})
          shuffled_accuracy <- shuffled_accuracy$results$RMSE

          accuracy_change <- original_accuracy - shuffled_accuracy

          placebo_temp <- data.frame(rep_count = rep_count, variable = variable, accuracy_change = accuracy_change)
          placebos <- bind_rows(placebos, placebo_temp)
        }

        if (rep_count %% 5){
          cat(paste0('           Completed Iteration ', rep_count, '\n'))
        }

      }

      return(placebos)

    }

    dropping_vars.bin <- function(mod.with, predictors, d.test){

      message("    Beginning Variable Omission Protocol...")

      fit_change <- data.frame()

      if (is.null(interactions)){
        all_combinations <- c(predictors)
        generate_combinations_df <- function(predictorss) {
          combinations <- list()

          # Generate combinations excluding one predictor at a time
          for (i in seq_along(predictors)) {
            current_predictors <- predictors[-i]
            current_combination <- c(current_predictors, interactions)
            combinations[[paste("Combination", i)]] <- data.frame(combination = paste(current_combination, collapse = ", "), dropped = predictors[i])
          }

          # Combine the dataframes into one
          result_df <- do.call(rbind, combinations)

          return(result_df)
        }
        combinations <- generate_combinations_df(predictors)

      } else {
        all_combinations <- c(predictors, interactions)
        generate_combinations_df <- function(predictors, interactions) {
          combinations <- list()

          # Generate combinations excluding one predictor at a time
          for (i in seq_along(predictors)) {
            current_predictors <- predictors[-i]
            current_combination <- c(current_predictors, interactions)
            combinations[[paste("Combination", i)]] <- data.frame(combination = paste(current_combination, collapse = ", "), dropped = predictors[i])
          }

          # Generate combinations excluding the interaction
          for (j in seq_along(interactions)) {
            current_interactions <- interactions[-j]
            current_combination <- c(predictors, current_interactions)
            combinations[[paste("Combination", j + length(predictors))]] <- data.frame(combination = paste(current_combination, collapse = ", "), dropped = interactions[j])
          }

          # Combine the dataframes into one
          result_df <- do.call(rbind, combinations)

          return(result_df)
        }
        combinations <- generate_combinations_df(predictors, interactions)

      }

      drop_combinations <- data.frame()

      for (i in 1:nrow(combinations)){
        temp_combination <- gsub("\\, ", " + ", combinations$combination[i])
        temp_combination <- paste0('as.factor(y) ~ ', temp_combination)
        dropped_var = combinations$dropped[i]
        drop_combinations <- bind_rows(drop_combinations, data.frame(combination = temp_combination, dropped_var = dropped_var))
      }


      for (c in 1:nrow(drop_combinations)){

        combination = drop_combinations$combination[c]
        dropped_var = drop_combinations$dropped_var[c]

        tc <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, savePredictions = TRUE)

        capture_output_mod.without_var <- capture.output({ mod.without_var <- suppressWarnings(
          train(as.formula(combination),
                method = as.character(ml_model),
                data= d.test,
                trControl = tc,
                importance=TRUE,
                localImp=TRUE))})

        RMSE_drop_var <- max(mod.without_var$results$Accuracy)
        original_RMSE <- max(mod.with$results$Accuracy)



        change <- data.frame(var = dropped_var, fit_change = original_RMSE - RMSE_drop_var)

        fit_change <- bind_rows(fit_change, change)

        cat(paste0('           Completed Var: ', dropped_var, '\n'))

      }

      return(fit_change)

    }
    dropping_vars.cont <- function(mod.with, predictors, d.test){

      message("    Beginning Variable Omission Protocol...")

      fit_change <- data.frame()

      if (is.null(interactions)){
        all_combinations <- c(predictors)
        generate_combinations_df <- function(predictorss) {
          combinations <- list()

          # Generate combinations excluding one predictor at a time
          for (i in seq_along(predictors)) {
            current_predictors <- predictors[-i]
            current_combination <- c(current_predictors, interactions)
            combinations[[paste("Combination", i)]] <- data.frame(combination = paste(current_combination, collapse = ", "), dropped = predictors[i])
          }

          # Combine the dataframes into one
          result_df <- do.call(rbind, combinations)

          return(result_df)
        }
        combinations <- generate_combinations_df(predictors)

      } else {
        all_combinations <- c(predictors, interactions)
        generate_combinations_df <- function(predictors, interactions) {
          combinations <- list()

          # Generate combinations excluding one predictor at a time
          for (i in seq_along(predictors)) {
            current_predictors <- predictors[-i]
            current_combination <- c(current_predictors, interactions)
            combinations[[paste("Combination", i)]] <- data.frame(combination = paste(current_combination, collapse = ", "), dropped = predictors[i])
          }

          # Generate combinations excluding the interaction
          for (j in seq_along(interactions)) {
            current_interactions <- interactions[-j]
            current_combination <- c(predictors, current_interactions)
            combinations[[paste("Combination", j + length(predictors))]] <- data.frame(combination = paste(current_combination, collapse = ", "), dropped = interactions[j])
          }

          # Combine the dataframes into one
          result_df <- do.call(rbind, combinations)

          return(result_df)
        }
        combinations <- generate_combinations_df(predictors, interactions)

      }

      drop_combinations <- data.frame()

      for (i in 1:nrow(combinations)){
        temp_combination <- gsub("\\, ", " + ", combinations$combination[i])
        temp_combination <- paste0('y ~ ', temp_combination)
        dropped_var = combinations$dropped[i]
        drop_combinations <- bind_rows(drop_combinations, data.frame(combination = temp_combination, dropped_var = dropped_var))
      }


      for (c in 1:nrow(drop_combinations)){

        combination = drop_combinations$combination[c]
        dropped_var = drop_combinations$dropped_var[c]

        tc <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, savePredictions = TRUE)

        capture_output_mod.without_var <- capture.output({ mod.without_var <- suppressWarnings(
          train(as.formula(combination),
                method = as.character(ml_model),
                data= d.test,
                trControl = tc,
                importance=TRUE,
                localImp=TRUE))})

        RMSE_drop_var <- max(mod.without_var$results$RMSE)
        original_RMSE <- max(mod.with$results$RMSE)



        change <- data.frame(var = dropped_var, fit_change = original_RMSE - RMSE_drop_var)

        fit_change <- bind_rows(fit_change, change)

        cat(paste0('           Completed Var: ', dropped_var, '\n'))

      }

      return(fit_change)

    }

    thisforest.bin <- function(y, predictors, Z = dat, ntrain = train.set) {
      set.seed(seed)
      registerDoParallel(numCores)
      Z <- Z[names(Z) %in% predictors]

      d <- cbind(y, Z)
      d <- as.data.frame(d)
      names(d)[1] <- 'y'
      d$y <- ifelse(d$y >= 1, 1, 0)

      d.train <- d[seq(ntrain),]
      d.test <- d[-seq(ntrain),]

      tc <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, savePredictions = TRUE)

      capture_output_mod.with <- capture.output({
        mod.with <- suppressWarnings(
          train(as.formula(full_vars),
                method = as.character(ml_model),
                data = d.train,
                trControl = tc,
                importance = TRUE,
                localImp = TRUE)
        )
      })

      capture_output_rf.basic <- capture.output({rf.basic <- randomForest(as.formula(full_vars),
                                                                          data=d.train,
                                                                          na.action="na.omit",
                                                                          ntree=1000,
                                                                          nodesize=4,
                                                                          importance=TRUE,
                                                                          localImp=TRUE)})


      placebo_base <- placebo_shuffle.bin(mod.with, d.train, d.test)

      placebo <- placebo_base %>%
        select(-rep_count) %>%
        rename(var = variable) %>%
        group_by(var) %>%
        summarize(min_change = quantile(accuracy_change, 0.025),
                  max_change = quantile(accuracy_change, 0.975))


      fit_change <- dropping_vars.bin(mod.with, predictors, d.test)

      fit_assess <- left_join(fit_change, placebo, by = 'var')


      olist <- list(
        with = mod.with,
        rf.basic = rf.basic,
        X = dat,
        with.test = d.test,
        with.train = d.train,
        var = predictors,
        training.data.with = d[seq(ntrain), ],
        kiv = d[-seq(ntrain), predictors, drop = FALSE],
        test.y = d[-seq(ntrain), ]$y,
        acc.ch = fit_assess,
        placebo = placebo_base
      )

      pusher <- push.bin(olist)
      olist$push = pusher

      return(olist)
    }
    thisforest.cont <- function(y, predictors, Z = dat, ntrain = train.set) {
      set.seed(seed)

      registerDoParallel(numCores)

      Z <- Z[names(Z) %in% predictors]

      d <- cbind(y, Z)
      d = as.data.frame(d)

      d.train <- d[seq(ntrain),]
      d.test <- d[-seq(ntrain),]

      tc <- trainControl(method = 'repeatedcv', number = 10, repeats = 3, savePredictions = TRUE)

      capture_output_mod.with <- capture.output({
        mod.with <- suppressWarnings(
          train(as.formula(full_vars),
                method = as.character(ml_model),
                data = d.train,
                trControl = tc,
                importance = TRUE,
                localImp = TRUE)
        )
      })

      capture_output_rf.basic <- capture.output({rf.basic <- randomForest(as.formula(full_vars),
                                                                          data=d.train,
                                                                          na.action="na.omit",
                                                                          ntree=1000,
                                                                          nodesize=4,
                                                                          importance=TRUE,
                                                                          localImp=TRUE)})


      placebo_base <- placebo_shuffle.cont(mod.with, d.train, d.test)

      placebo <- placebo_base %>%
        select(-rep_count) %>%
        rename(var = variable) %>%
        group_by(var) %>%
        summarize(min_change = quantile(accuracy_change, 0.025),
                  max_change = quantile(accuracy_change, 0.975))


      fit_change <- dropping_vars.cont(mod.with, predictors, d.test)

      fit_assess <- left_join(fit_change, placebo, by = 'var')


      olist <- list(
        with = mod.with,
        rf.basic = rf.basic,
        X = dat,
        with.test = d.test,
        with.train = d.train,
        var = predictors,
        training.data.with = d[seq(ntrain), ],
        kiv = d[-seq(ntrain), predictors, drop = FALSE],
        test.y = d[-seq(ntrain), ]$y,
        acc.ch = fit_assess,
        placebo = placebo_base
      )

      pusher <- push.cont(olist)
      olist$push = pusher

      return(olist)
    }

  } #Routine Functions

  if (data_type == 'Binomial'){
    sandbox.models$pai <- thisforest.bin(y = outcome_var, predictors = predictors)
  } else {
    sandbox.models$pai <- thisforest.cont(y = outcome_var, predictors = predictors)
  } #Implementing by Data Type


  message("-------------------------------------------------------------------")
  cat("-------------------------- PAI  Complete --------------------------\n")
  message("-------------------------------------------------------------------")

  return(sandbox.models)


}





################################################################################
#Test w/ Sample Data
################################################################################

set.seed(1234)
test_data <- data.frame(y = rbinom(20, 1, 0.5),
                   var1 = rnorm(1000, mean = 15, sd = 1),
                   var2 = rnorm(1000, mean = 10, sd = 1),
                   var3 = rnorm(1000, mean = 5, sd = 2),
                   var4 = rnorm(1000, mean = 30, sd = 1))


test <- pai_main(data = test_data,
                 outcome = "y",
                 predictors = c("var1", "var2", "var3"),
                 interactions = c("var1*var2"),
                 ml = c("rf", 8, 1),
                 seed = 1234)


data = test_data
outcome = "y"
predictors = c("var1", "var2", "var3")
interactions = c("var1*var2")
ml = c("rf", 8, 1)
seed = 1234
################################################################################
#Plot Sample Output
################################################################################


pai_plot_BASE_flexible <- function(data,
                                   plot_type = NULL, #Placebo or Steps
                                   variables = NULL, #Variables to Declare (Null Default = All)
                                   plot_points = FALSE) {

  data <- data$pai

  if(is.null(variables)){
    variables = data$var
  }

  if(is.null(plot_type)){
    plot_type <- 'Placebo'
  } # Assign 'Placebo' if plot_type = NULL

  if(plot_type == 'Placebo'){

    {
      temp <- data$acc.ch %>%
        mutate(var_numeric = 1:nrow(data$acc.ch)) %>%
        mutate(var = ifelse(grepl("\\*", var), gsub("\\*", " x\n", var), var))


      temp_figure <- ggplot(data = temp, aes(x = var_numeric, y = fit_change)) +
        geom_rect(aes(xmin = var_numeric - 0.15, xmax = var_numeric + 0.15,
                      ymin = min_change, ymax = max_change, fill = 'Range of Predicted Accuracy from Placebos'),
                  color = 'black') +
        geom_point(aes(color = 'Prediction from Model Fit After Dropping Information'), size = 2.5) +
        geom_hline(yintercept = 0, linetype = 2) +
        scale_fill_manual(values = 'gray', name = NULL) +
        scale_color_manual(values = 'gray5', name = NULL) +
        scale_x_continuous(breaks = seq(min(temp$var_numeric), max(temp$var_numeric), 1), labels = temp$var) +
        theme_minimal() +
        theme_test(base_size = 14) +
        labs(
          x = '\n',
          y = 'Change in Predicted Accuracy\nWith All True Data\n') +
        theme(
          axis.text = element_text(size = 14),
          panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
          legend.title.align = 0.5,
          legend.text.align = 0.25,
          legend.title = element_blank(),
          legend.text = element_text(size = 12, color = "gray5"),
          legend.position = "bottom",
          strip.text = element_text(size = 14, face = "bold"),
          strip.background = element_rect(fill = "gray", color = "gray5"),
          plot.title = element_text(size = 18, face = "bold"),
          plot.subtitle = element_text(size = 15),
          plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))

      return(temp_figure)

    } #Return Figure



  }

  else if(plot_type == 'Steps'){

    figures <- list()

    for (var in variables){
      temp <- data.frame(data$push[var])
      names(temp) <- gsub(paste0(var, "\\."), '', names(temp))

      temp_data <- tidyr::gather(temp, key = "variable", value = "value", -steps)

      temp_figure <- ggplot(data = temp_data, aes(x = steps, y = value)) +
        stat_smooth(method = 'lm', geom = 'smooth', formula = y ~ x, se = FALSE, size = 1, colour = 'deepskyblue3') +
        theme_minimal() +
        geom_hline(yintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/3) +
        geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/3) +
        labs(title = var) +
        theme(
          axis.text = element_text(size = 14),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
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
          plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))



      if (plot_points == TRUE){
        temp_figure <- temp_figure + geom_point(alpha = 1/10)
      }


      figures[[var]] <- temp_figure

    }
    yleft <- textGrob("\nEffect on Outcome\n", rot = 90, gp = gpar(fontsize = 15))
    bottom <- textGrob("\nSteps\n",  gp = gpar(fontsize = 15))
    uni <-  grid.arrange(grobs = figures, left = yleft, bottom = bottom)
    return(uni)
  }


}



pai_plot_BASE_flexible(test,
                       plot_type = 'Placebo',
                       plot_points = FALSE)


################################################################################
#Diagnostic Tool
################################################################################

pai_diagnostic <- function(pai_object = NULL,
                           variables = NULL,
                           bins = 10){

  data = pai_object$pai$push

  figure_list <- list()

  if (is.null(variables)){
    variables = names(pai_object$pai$push)
  }


  for (var in variables){

    temp_dat <- data.frame(data[[var]])
    temp_dat <- tidyr::pivot_longer(temp_dat, cols = -steps, names_to = "variable", values_to = "value")
    temp_dat$steps <- as.numeric(temp_dat$steps)
    num_bins = bins
    temp_dat$bin <- cut_interval(temp_dat$steps, n = num_bins)

    temp_dat <- temp_dat %>%
      rowwise() %>%
      mutate(cut = ifelse(is.numeric(as.numeric(gsub(".*\\,", "", gsub("\\]", "", bin)))),
                          gsub(".*\\,", "", gsub("\\]", "", bin)),
                          0)) %>%
      mutate(cut = as.numeric(cut))

    get_slope <- function(data) {
      lm_fit <- lm(value ~ steps, data = data)
      return(coef(lm_fit)[2])
    }
    slope_values <- data.frame(
      bin = unique(temp_dat$bin),
      slope = sapply(unique(temp_dat$bin), function(bin) get_slope(temp_dat[temp_dat$bin == bin, ]))
    )
    slope_values <- slope_values %>%
      mutate(slope = round(slope, 3)) %>%
      rename(`Steps Bin` = bin,
             Slope = slope)

    temp_figure <- ggplot(temp_dat, aes(x = steps, y = value)) +
      geom_point(colour = 'gray5', alpha = 1/8) +
      stat_smooth(method = "lm", se = FALSE, aes(color = bin), size = 1.25) +
      theme_minimal() +
      labs(
        title = var,
        x = '\nSteps',
        y = ' '
      ) +
      geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
      theme(
        axis.text = element_text(size = 14),
        axis.title = element_text(size = 16),
        panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
        panel.grid = element_blank(),
        legend.title.align = 0.5,
        legend.text.align = 0.25,
        legend.title = element_blank(),
        legend.text = element_text(size = 15, color = "gray5"),
        legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
        legend.position = "none",
        strip.text = element_text(size = 14, face = "bold"),
        strip.background = element_rect(fill = "gray", color = "gray5"),
        plot.title = element_text(size = 18, face = "bold"),
        plot.subtitle = element_text(size = 15),
        plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))


    figure_list$Figures[[var]] <- temp_figure
    figure_list$Slopes[[var]] <- slope_values

  }

  return(figure_list)

}

c <- pai_diagnostic(pai_object = test,
                    bins = 15,
                    variables = NULL)



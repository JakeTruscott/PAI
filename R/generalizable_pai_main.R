################################################################################
#Prediction as Inference (PAI) Package - R
#Updating Sandbox Materials
#Code Developed by Ben Johnson (UF), Logan Strother (Purdue), and Jake Truscott (Purdue)
#Updated January 2024
#Contact: jtruscot@purdue.edu
################################################################################

################################################################################
#Load Packages
################################################################################
library(randomForest); library(doParallel); library(caret); library(parallel); library(rlist); library(dplyr); library(gridExtra); library(gridtext); library(grid); library(doSNOW); library(patchwork)


################################################################################
# Generalizable PAI Main Function
################################################################################
'
#data -- Dataframe

#outcome -- Single Character from #data (Required)

#predictors -- Character or Character Vector (Required) ~ Default to ALL Non-"outcome"

#interactions -- Character Vector (Optional - Default to Null -- Indicate as Single Character with term(s) separated by "*" ~  Ex: "var1*var2")

#drop_vars #Character Vector (Optional - Default to ALL)

# ml:
  [1] ML Model (caret R)
  [2] Parallel Cores (Default to 1)
  [3] Placebo Iterations (Default 10)
  [4] K-Folds (Default 10)

# additional_tC = Additional Train Control Params

# Seed -- Random Seed Generator (Default to 1234)
'

'
Notes:
1) Make sure to reference caret package documentation for trainControl-specific parameters & have libraries downloaded -- Will register "Selection" to download non-installed packages needed for specific ML Models (ex: "fastAdaboost" for boosted trees)

'

set.seed(1234)
test_data <- data.frame(y = sample(0:1, 1000, replace = TRUE),
                        var1 = rnorm(1000, mean = 15, sd = 1),
                        var2 = rnorm(1000, mean = 10, sd = 1),
                        var3 = rnorm(1000, mean = 5, sd = 2),
                        var4 = rnorm(1000, mean = 30, sd = 1))


pai_main <- function(data,
                     outcome = NULL,
                     predictors = NULL,
                     interactions = NULL,
                     drop_vars = NULL,
                     ml = c(NA, NA, 10, 10),
                     custom_tc = "method = 'repeatedcv', number = cv_folds, repeats = 2, savePredictions = TRUE",
                     seed = 1234){

  message("-------------------------------------------------------------------")
  cat("---------------------- Beginning PAI Process ----------------------\n")
  message("-------------------------------------------------------------------")

  {

    parameters <- list()

    if (is.null(outcome)){
      stop('No Outcome Variable Declared \n Declare Outome & Try Again')
    } else {
      parameters['outcome'] <- outcome
    } # outcome


    if (is.null(predictors)){
      parameters['predictors'] <- names(data[!names(data) %in% outcome])
    } else {
      parameters['predictors'] <- paste0(predictors, collapse = ", ")
    } #predictors

    if (is.null(interactions) || length(interactions) == 0) {
      parameters['interactions'] <- 'None'
    } else {
      # Use paste to concatenate elements with a space separator
      interactions_str <- paste(interactions, collapse = ' ')

      # Ensure that the length of the vector matches the length of the element
      parameters['interactions'] <- rep(interactions_str, length.out = length(parameters['interactions']))
    }
    #interactions

    if (is.null(drop_vars)){
      parameters['drop_vars'] <- 'All Predictors'
    } else {
      parameters['drop_vars'] <- paste(drop_vars, collapse = ", ")
    } #drop_vars

    if (is.na(ml[1])){
      parameters['ml_model'] <- 'rf'
    } else {
      parameters['ml_model'] <- ml[1]
    } #ML Model (caret R ~ Default to Random Forest)

    if (is.na(ml[2])){
      parameters['cores'] <- 1
    } else {
      parameters['cores'] <- as.numeric(ml[2])
    } # Cores (Default to 1)

    if (is.na(ml[3])){
      parameters['placebo_iterations'] <- 10
    } else {
      parameters['placebo_iterations'] <- as.numeric(ml[3])
    } #placebo iterations

    if (is.na(ml[4])){
      parameters['k_folds'] <- 10
    } else {
      parameters['k_folds'] <- as.numeric(ml[4])
    } #k_folds

    if (length(unique(data[[outcome]])) == 2){
      parameters['data_type'] = 'Binomial'
    } else {
      parameters['data_type'] = 'Continuous'
    }

    if (is.null(custom_tc)){
      parameters['custom_tc'] <- 'FALSE'
    } else {
      parameters['custom_tc'] <- 'TRUE'
    }

    if (is.null(seed)){
      parameters['seed'] <- 1234
    } else {
      parameters['seed'] <- as.numeric(seed)
    }

    message(
      "    Dependent Variable = ", parameters$outcome, "\n",
      "    Data Type = ", parameters$data_type, "\n",
      "    Predictors = ", parameters$predictors, "\n",
      "    Interactions = ", parameters$interactions, "\n",
      "    Variables to Iteratively Drop = ", parameters$drop_vars, "\n",
      "    Custom Train Control = ", parameters$custom_tc, "\n",
      "    Seed = ", parameters$seed, "\n",
      '    ML Parameters:', "\n",
      "          ML Model: ", parameters$ml_model, "\n",
      "          Cores: ", parameters$cores, "\n",
      "          Placebo Shuffling Repetitions: ", parameters$placebo_iterations, "\n",
      "          Cross-Validation Folds: ", parameters$k_folds)


  } #Print Parameters
  {

    interaction_terms <- c()
    if(length(interactions) >= 1){
      for (interaction in 1:length(interactions)){
        interaction_temp <- unlist(stringr::str_split(interactions[interaction], pattern = '\\*'))
        interaction_temp <- paste(interaction_temp, collapse = ":")
        interaction_terms <- c(interaction_terms, interaction_temp)

      }
    } else {
      interaction_terms = NULL
    } #Incorporate Interaction Terms (In Case Terms Declared in Interaction But Not Predictors)

    data_type = parameters$data_type
    variables = unlist(stringr::str_split(parameters$predictors, pattern = '\\,'))
    if(data_type == 'Binomial'){
      formula = paste("as.factor(y) ~", paste(variables, collapse = " + "))
      if (is.null(interaction_terms)){
        formula = as.formula(formula)
      } else {
        formula = paste0(formula, " + ", paste(interaction_terms, collapse = " + "))
        formula = as.formula(formula)
      }

    } else {
      formula = paste("y ~", paste(variables, collapse = " + "))
      if (is.null(interaction_terms)){
        formula = as.formula(formula)
      } else {
        formula = paste0(formula, " + ", paste(interaction_terms, collapse = " + "))
        formula = as.formula(formula)
      }
    }

    dat <- data %>%
      select(any_of(c(outcome, predictors)))

    if(is.null(interaction_terms)){
      dat <- dat
    } else {
      interaction_cols <- list()
      for (int in 1:length(interaction_terms)){
        temp_in <- unlist(stringr::str_split(interaction_terms[int], pattern = '\\:'))
        temp_in_cols <- data %>%
          select(any_of(temp_in)) %>%
          mutate(interaction = rowSums(across(everything(), .fns = ~. * .)))
        interaction_cols[[interaction_terms[int]]] <- temp_in_cols$interaction
      }
      interaction_cols <- data.frame(interaction_cols)
      names(interaction_cols) <- gsub('\\.', '\\:', names(interaction_cols))
      dat <- cbind(dat, interaction_cols)
    }




    pai_output <- list() #Declare List Object to Store Outputs

    ml_model <- parameters$ml_model
    placebo_iterations <- parameters$placebo_iterations
    train.set <- round(length(dat[[outcome]])/5, 0)
    outcome_var <- dat[outcome]
    cv_folds <- parameters$k_folds

    if (parameters$custom_tc == 'FALSE'){
      tc_main <- trainControl(method = 'repeatedcv',
                              number = cv_folds,
                              repeats = 3,
                              savePredictions = TRUE)
    } else {

      custom_params <- data.frame(custom_declare = strsplit(custom_tc, ', ')[[1]])

      tc <- list()

      tc_names <- names(trainControl())


      custom_params <- custom_params %>%
        mutate(param = gsub('\\=.*', '', custom_declare),
               value = gsub('.*\\=', '', custom_declare)) %>%
        mutate(param = gsub('\\s+', '', param),
               value = gsub('\\s+', '', value)) %>%
        select(param, value) %>%
        filter(param %in% tc_names)

      tc_params <- list()

      for (i in 1:nrow(custom_params)) {
        param_name <- custom_params$param[i]
        param_value <- eval(parse(text = custom_params$value[i]))
        tc_params[[param_name]] <- param_value
      }

      tc_main <- do.call(trainControl, tc_params)


    }


  } #Assign Additional Params for Functions
  {

    runpred <- function(mod, var, stepper, dat){

      if (data_type == 'Continuous'){
        dat[[var]] <- dat[[var]] + stepper
        pred <- predict(mod, dat)
        true <- dat$y
        dif <- pred - true
        return(dif)
      } else {
        dat[[var]] <- dat[[var]] + stepper
        pred <- predict(mod, dat)
        true <- dat$y
        onecount <- length(which(pred=='1'))/length(true)
        acc <- length(which(pred==true))/length(true)
        return(c(onecount, acc))
      }



    } #runpred

    push <- function(l){

      message("    Beginning Push Protocol...")

      push_output <- list()

      for (variable in 1:length(l$var)){

        x = l$var[variable]
        Z = l$with.test
        sdx <- sd(Z[,x])
        steps <- seq(-2*sdx, 2*sdx, (4*sdx)/100)
        tester <- lapply(steps, function(z) runpred(l$with, x, z, Z ))
        runpred_output <- cbind(steps, t(list.cbind(tester)))
        push_output[[x]] <- runpred_output
      }

      return(push_output)


    } #push

    placebo_shuffle <- function(w, d.train, d.test){

      message("    Beginning Placebo Protocol...")

      placebos <- data.frame() #Initialize Empty DF
      variables <- c(setdiff(names(d.test), "y"), interaction_terms) #Set Variables to Shuffle
      variables <- unique(variables)

      for (rep in 1:as.numeric(parameters$placebo_iterations)){
        for (variable in variables){

          capture_output_original_accuracy <- capture.output({
            suppressWarnings(original_accuracy <- train(form = formula,
                                                        data = d.test,
                                                        metric = ifelse(parameters$data_type == 'Continuous', 'RMSE', 'Accuracy'),
                                                        method = as.character(parameters$ml_model),
                                                        trControl = tc_main))
          }) #Get Original Accuracy

          if(data_type == 'Continuous'){
            original_accuracy <- original_accuracy$results$RMSE
          } else {
            original_accuracy <- original_accuracy$results$Accuracy
          } #Get Accuracy (Binomial) or RMSE (Continuous) Output


          shuffle_data <- d.test
          unique_interaction_vars <- unlist(stringr::str_split(parameters$interactions, pattern = '\\*')) #Get Unique Interaction Vars

          if (is.null(parameters$interactions)){
            shuffle_data <- d.test
          } else {

            if (variable %in% interaction_terms){
              shuffle_type = 'interaction'
            } else if (variable %in% unique_interaction_vars) {
              shuffle_type = 'interaction_contributor'
            } else {
              shuffle_type = 'non_interaction'
            } #Get Variable Type

            if (shuffle_type == 'non_contributor'){
              shuffle_data[[variable]] <- sample(shuffle_data[[variable]])
            } #If Variable Doesn't Contribute to (or is not) an Interaction

            else if (shuffle_type == 'interaction'){
              interaction_vars <- unlist(stringr::str_split(variable, pattern = '\\:')) #Get Vars in Interaction Term
              for (temp_var in interaction_vars){
                shuffle_data[[temp_var]] <- sample(shuffle_data[[temp_var]]) #Shuffle Vars Constituting Interaction
              }
              shuffled_columns <- shuffle_data[names(shuffle_data) %in% interaction_vars] #Get Shuffled Interaction Variable Cols
              shuffled_interaction <- shuffled_columns %>%
                mutate(interaction = rowSums(across(everything(), .fns = ~. * .))) %>%
                select(interaction) #Recreate the Interaction
              shuffle_data[variable] = shuffled_interaction$interaction #Put New Shuffled Interaction Back in Shuffle Data
            } # If Variable Is An Interaction Term

            else {
              shuffle_data[[variable]] <- sample(shuffle_data[[variable]]) #Shuffle Variable

              ints <- unlist(parameters$interactions) #Get Interaction Terms

              for (int in 1:length(ints)){
                column <- gsub('\\*', '\\:', ints[int])
                shuffled_columns  <- unlist(stringr::str_split(ints[int], pattern = '\\*'))
                shuffled_columns  <- shuffle_data[names(shuffle_data) %in% shuffled_columns ]
                shuffled_interaction <- shuffled_columns %>%
                  mutate(interaction = rowSums(across(everything(), .fns = ~. * .))) %>%
                  select(interaction)
                shuffle_data[column] = shuffled_interaction$interaction
              }

            } #If Variable Is Part of an Interaction Term



          } # Note: If Shuffling Var in Interaction (Even if Not Interaction Directly) -- Need to Shuffle Interaction Bc Downwind Effect of Shuffling Vars Contributing to Interaction...


          capture_output_shuffled_accuracy <- capture.output({
            suppressWarnings(shuffled_accuracy <- train(form = formula,
                                                        data = shuffle_data,
                                                        metric = ifelse(parameters$data_type == 'Continuous', 'RMSE', 'Accuracy'),
                                                        method = as.character(parameters$ml_model),
                                                        trControl = tc_main)) #Run Shuffled Accuracy
          })


          if(data_type == 'Continuous'){
            shuffled_accuracy <- shuffled_accuracy$results$RMSE
          } else {
            shuffled_accuracy <- shuffled_accuracy$results$Accuracy
          } # Get Accuracy or RMS Output

          accuracy_change <- as.numeric(mean(original_accuracy)) - as.numeric(mean(shuffled_accuracy)) #Calculate Accuracy Change

          placebo_temp <- data.frame(rep_count = rep, variable = variable, accuracy_change = accuracy_change)
          placebos <- bind_rows(placebos, placebo_temp)
        }

        if (rep %% 2 == 0 ){
          cat(paste0('           Completed Iteration ', rep, '\n'))
        }

      }

      return(placebos)


    } #placebo shuffling

    dropping_vars <- function(mod.with, d.test){

      message("    Beginning Variable Omission Protocol...")

      fit_change <- data.frame() #Initialize Empty DF for Fit Changes by Drop Var

      if (is.null(parameters$drop_vars)){
        drop <- names(d.test[!names(d.test) %in% 'y'])
      } else if (parameters$drop == 'All Predictors'){
        drop <- names(d.test[!names(d.test) %in% 'y'])
      } else {
        drop <- unique(parameters$drop_vars)
      } #Declare Dropvars by DropVar Type

      generate_combinations_df <- function(drop) {

        combinations <- list()

        for (i in seq_along(drop)){
          current_predictors <- drop[-i]
          combination_temp <- c(current_predictors)
          combinations[[paste('Combination_', i)]] <- data.frame(combination = paste(combination_temp, collapse = ", "), dropped = drop[i])
        } #Create List of Combinations -- Dropping 1 Var Each Time from drop_vars (drop)

        combinations <- do.call(rbind, combinations) #Combine Into Single DF

        return(combinations)
      } #Generate DropVar Combinations

      combinations <- generate_combinations_df(drop)  #Run generation_combinations_df

      drop_combinations <- data.frame() #Initialize Empty DF for Drop Variables Output

      for (i in 1:nrow(combinations)){
        temp_combination <- gsub('\\,', ' + ', combinations$combination[i])

        if (data_type == 'Continuous'){
          temp_combination <- paste0('y ~ ', temp_combination)
        } else {
          temp_combination <- paste0('as.factor(y) ~ ', temp_combination)
        } #Create Function Text - Exception for Data Type

        dropped_var <- combinations$dropped[i] #Get Dropped Var

        drop_combinations <- bind_rows(drop_combinations, data.frame(combination = temp_combination,
                                                                     dropped_var = dropped_var))
      } #Convert Drop Combinations to Formula Structure

      for (c in 1:nrow(drop_combinations)){

        combination = drop_combinations$combination[c]
        dropped_var = drop_combinations$dropped_var[c]

        capture_output_mod.with <- capture.output({ mod.without_var <- suppressWarnings( train(form = as.formula(combination),
                                                                                               data = d.test,
                                                                                               metric = ifelse(parameters$data_type == 'Continuous', 'RMSE', 'Accuracy'),
                                                                                               method = as.character(parameters$ml_model),
                                                                                               trControl = tc_main)
        )})

        if (data_type == 'Continuous'){
          fit_drop_var <- mean(mod.without_var$results$RMSE)
          fit_original <- mean(mod.with$results$RMSE)
        } else {
          fit_drop_var <- mean(mod.without_var$results$Accuracy)
          fit_original <- mean(mod.with$results$Accuracy)
        } # Get Fit -- Exception by Data Type

        change_temp <- data.frame(var = dropped_var,
                                  fit_change = (fit_original - fit_drop_var)) #Get Temp Frame for Fit Change

        fit_change <- bind_rows(fit_change, change_temp) #Append to fit_change

        cat(paste0('           Completed Var: ', dropped_var, '\n'))

      }

      return(fit_change) #Return Full DF When Done

    } #dropping vars iteratively

    pai_ml <- function(y, full_dat, ntrain, predictors, interactions){

      set.seed(seed) #Set Random Seed

      registerDoParallel(as.numeric(parameters$cores)) #Register Parallel Environment

      d <- full_dat[names(full_dat) %in% c(predictors, gsub('\\*', '\\:', interactions))]
      d <- cbind(y, d)
      names(d)[1] <- 'y'
      if (data_type == 'Binomial'){
        d$y <- as.factor(d$y)
      } #Create DF for Train/Test Split

      d.train <- d[seq(ntrain),] #Set Training Data
      d.test <- d[-seq(ntrain),] # Set Testing Data

      capture_output_mod.with <- capture.output({

        suppressWarnings(

          mod.with <- train(form = formula,
                            data = d.train,
                            metric = ifelse(parameters$data_type == 'Continuous', 'RMSE', 'Accuracy'),
                            method = as.character(parameters$ml_model),
                            trControl = tc_main,
                            localImp = TRUE)


        )
      }) #Get mod.with

      tc_rf <- trainControl(method = 'repeatedcv', number = cv_folds, repeats = 3, savePredictions = TRUE) #TC for Basic RF
      capture_output_rf.basic <- capture.output({

        suppressWarnings(

          rf.basic <- train(form = formula,
                            data = d.train,
                            metric = ifelse(parameters$data_type == 'Continuous', 'RMSE', 'Accuracy'),
                            method = 'rf',
                            trControl = tc_rf,
                            localImp = TRUE)

        )
      }) #Get Basic RF

      placebo_base <- placebo_shuffle(mod.with, d.train, d.test) #Initialize Placebo_Shuffle
      placebo <- placebo_base %>%
        select(-rep_count) %>%
        rename(var = variable) %>%
        group_by(var) %>%
        summarize(min_change = quantile(accuracy_change, 0.025),
                  max_change = quantile(accuracy_change, 0.975)) #Post Process Placebo_Shuffle

      fit_change <- dropping_vars(mod.with, d.test) #Initialize Dropping_Vars
      fit_assess <- left_join(fit_change, placebo, by = 'var') #Join Fit Changes by Dropped Var ('var')

      output_list <- list(
        input_parameters = parameters,
        with = mod.with,
        rf.basic = rf.basic,
        X = full_dat,
        with.test = d.test,
        with.train = d.train,
        var = c(predictors, gsub('\\*', '\\:', interactions)),
        drop_acc.ch = fit_assess,
        placebo = placebo) #Compile All Into Single List Object

      pusher <- push(output_list) #Initialize push
      output_list$push = pusher #Put push output in output_list

      return(output_list)



    } #Main Function



  } #Core Functions

  pai_output <- pai_ml(y = outcome_var, full_dat = dat, ntrain = train.set, predictors = predictors, interactions = interactions) #Initialize Core Functions

  message("-------------------------------------------------------------------")
  cat("-------------------------- PAI  Complete --------------------------\n")
  message("-------------------------------------------------------------------")

  return(pai_output)



}

test <- pai_main(data = test_data,
                 outcome = 'y',
                 predictors = c('var1', 'var2', 'var3'),
                 interactions = 'var1*var2',
                 drop_vars = NULL,
                 ml = c('parRF', 8, 5, 10),
                 custom_tc = "method = 'repeatedcv', number = cv_folds, repeats = 2, savePredictions = TRUE",
                 seed = 1234)



################################################################################
# PAI Diagnostic Tool
################################################################################


pai_diagnostic <- function(pai_object = NULL,
                           variables = NULL,
                           bins = 10,
                           bin_cut = NULL){

  diagnostic_list <- list()
  combined_diagnostic_list <- list()

  {

    if (is.null(variables)){
      variables = unique(pai_object$var)
      variables = variables[!grepl('\\:', variables)]
    }

    if (is.null(bins)){
      bins = 5
    }

    if (is.null(bin_cut)){
      bin_cut = 5
    }

  } #Set Parameters

  {

    temp <- pai_object$drop_acc.ch %>%
      mutate(var_numeric = 1:nrow(pai_object$drop_acc.ch)) %>%
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
        plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
        plot.subtitle = element_text(size = 15),
        plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))

    diagnostic_list$figures[['placebo']] <- temp_figure


  } #Placebo

  {

    data <- pai_object$push

    for (var in variables){

      temp_dat <- data.frame(data[[var]])
      temp_dat <- temp_dat[,c(1,3)]
      names(temp_dat) <- c('steps', 'accuracy')
      temp_dat$steps <- as.numeric(temp_dat$steps)

      lm_temp <- lm(accuracy ~ steps, data = temp_dat)
      ci_temp <- predict(lm_temp, interval = "confidence")
      temp_dat <- cbind(temp_dat, ci_temp)

      summary_lm <- suppressWarnings(summary(lm_temp))
      slope <- coef(summary_lm)[2]
      se_slope <- summary_lm$coefficients[2, "Std. Error"]
      p_value <- summary_lm$coefficients[2, "Pr(>|t|)"]

      slope_frame <- data.frame(
        slope = round(slope, 3),
        se = round(se_slope, 3),
        p_value = round(p_value, 5)) %>%
        mutate(p_value = case_when(
          .default = "",
          p_value < 0.05 & p_value >= 0.01 ~ "*",
          p_value < 0.01 & p_value >= 0.001 ~ '**',
          p_value < 0.001 ~ '***'
        )) %>%
        unique()


      {
        temp_figure_stand_alone <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)'), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = lwr, ymax = upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = var,
            x = paste("\nSteps\n(", expression("\U00B1 2 \U03C3"), ")"),
            y = 'Accuracy\n',
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
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
            legend.position = "bottom",
            strip.text = element_text(size = 14, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))
        } #Stand Alone

      {
        temp_figure_combined <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)'), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = lwr, ymax = upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = 'Linear Fit',
            x = "\n",
            y = '\n',
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
          theme(
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 10, face = 'bold'),
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
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))


        temp_figure

      } #Combined Diagnostic

      diagnostic_list$figures$linear[[var]] <- temp_figure_stand_alone
      diagnostic_list$slope_tables$linear[[var]] <- slope_frame
      combined_diagnostic_list$figures$linear[[var]] <- temp_figure_combined

    }


  } #Full Linear Fit

  {

    data <- pai_object$push

    for (var in variables){

      temp_dat <- data.frame(data[[var]])
      temp_dat <- temp_dat[,c(1,3)]
      names(temp_dat) <- c('steps', 'accuracy')

      temp_dat$steps <- as.numeric(temp_dat$steps)
      num_bins = bins
      temp_dat$bin <- cut_interval(temp_dat$steps, n = num_bins)

      temp_dat <- temp_dat %>%
        rowwise() %>%
        mutate(cut = ifelse(is.numeric(as.numeric(gsub(".*\\,", "", gsub("\\]", "", bin)))),
                            gsub(".*\\,", "", gsub("\\]", "", bin)),
                            0)) %>%
        mutate(cut = as.numeric(cut))

      {
        ci_temp <- data.frame()
        slope_temp <- data.frame()
        unique_bins <- unique(temp_dat$bin)

        for (b in unique_bins){

          temp_bin <- temp_dat[temp_dat$bin == b,]
          lm_bin_temp <- lm(accuracy ~ steps, data = temp_bin)

          summary_lm <- summary(lm_bin_temp)
          slope <- coef(summary_lm)[2]
          se_slope <- summary_lm$coefficients[2, "Std. Error"]
          p_value <- summary_lm$coefficients[2, "Pr(>|t|)"]

          slope_frame <- data.frame(
            bin = b,
            slope = round(slope, 3),
            se = round(se_slope, 3),
            p_value = round(p_value, 5)) %>%
            mutate(sig = case_when(
              .default = "",
              p_value < 0.05 & p_value >= 0.01 ~ "*",
              p_value < 0.01 & p_value >= 0.001 ~ '**',
              p_value < 0.001 ~ '***'
            )) %>%
            unique()

          slope_temp <- bind_rows(slope_temp, slope_frame)

          ci_bin_temp <- predict(lm_bin_temp, interval = 'confidence')
          ci_bin_temp <- data.frame(ci = ci_bin_temp)


          ci_temp <- bind_rows(ci_temp, ci_bin_temp)

        }

        temp_dat <- cbind(temp_dat, ci_temp)

        } #Slopes

      {

        temp_figure_stand_alone <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)', group = bin), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = ci.lwr, ymax = ci.upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = var,
            x = paste("\nSteps\n(", expression("\U00B1 2 \U03C3"), ")"),
            y = 'Accuracy\n',
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
          theme(
            axis.text = element_text(size = 14),
            axis.title = element_text(size = 16, face = 'bold'),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            panel.grid = element_blank(),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "bottom",
            strip.text = element_text(size = 14, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))


      } #Stand Alone Figure

      {
        temp_figure_combined <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)', group = bin), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = ci.lwr, ymax = ci.upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = 'Static Bins',
            x = "\n",
            y = "\n",
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
          theme(
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 10, face = 'bold'),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            panel.grid = element_blank(),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "none",
            strip.text = element_text(size = 10, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))


      } #Combined

      diagnostic_list$figures$static[[var]] <- temp_figure_stand_alone
      combined_diagnostic_list$figures$static[[var]] <- temp_figure_combined
      diagnostic_list$slope_tables$static[[var]] <- slope_temp

    }


  } #Static Bins

  {

    data <- pai_object$push

    for (var in variables){

      temp_dat <- data.frame(data[[var]])
      temp_dat <- temp_dat[,c(1,3)]
      names(temp_dat) <- c('steps', 'accuracy')

      temp_dat$steps <- as.numeric(temp_dat$steps)
      num_bins = bins
      temp_dat$bin <- cut_interval(temp_dat$steps, n = num_bins)

      temp_dat <- temp_dat %>%
        rowwise() %>%
        mutate(cut = ifelse(is.numeric(as.numeric(gsub(".*\\,", "", gsub("\\]", "", bin)))),
                            gsub(".*\\,", "", gsub("\\]", "", bin)),
                            0)) %>%
        mutate(cut = as.numeric(cut)) %>%
        group_by(cut) %>%
        mutate(bin_id = cur_group_id())

      {
        t <- data.frame()
        bin_cut_id <- 1
        for (i in unique(temp_dat$bin_id)){
          max_bin <- max(temp_dat$bin_id)
          bin_1 <- i
          bin_2 <- ifelse(i == max_bin, i, bin_1 + 1)
          bin_1_dat <- temp_dat %>%
            filter(bin_id == bin_1)
          bin_2_dat <- temp_dat %>%
            filter(bin_id == bin_2)

          if (i == max_bin){
            temp_bin_dat <- bin_1_dat
            temp_bin_dat$bin_cut_id <- bin_cut_id
          } else {
            temp_bin_dat <- bind_rows(bin_1_dat, bin_2_dat)
            temp_bin_dat$bin_cut_id <- bin_cut_id
          }

          t <- bind_rows(t, temp_bin_dat)

          bin_cut_id <- bin_cut_id + 1

        }

        bin_cut_match <- t %>%
          select(steps, bin_cut_id) %>%
          group_by(bin_cut_id) %>%
          mutate(bin_range = paste0("(", round(min(steps), 3), ", ", round(max(steps), 3), "]")) %>%
          mutate(bin_group = paste0(bin_range, ' Bin (', bin_cut_id, ') \n Slope = '))

        ci_temp <- data.frame()
        slope_temp <- data.frame()
        unique_bins <- unique(t$bin_cut_id)

        for (b in unique_bins){

          temp_bin <- t[t$bin_cut_id == b,]
          lm_bin_temp <- lm(accuracy ~ steps, data = temp_bin)

          summary_lm <- summary(lm_bin_temp)
          slope <- coef(summary_lm)[2]
          se_slope <- summary_lm$coefficients[2, "Std. Error"]
          p_value <- summary_lm$coefficients[2, "Pr(>|t|)"]

          slope_frame <- data.frame(
            bin = bin_cut_match$bin_range[match(b, bin_cut_match$bin_cut_id)],
            slope = round(slope, 3),
            se = round(se_slope, 3),
            p_value = round(p_value, 5)) %>%
            mutate(sig = case_when(
              .default = "",
              p_value < 0.05 & p_value >= 0.01 ~ "*",
              p_value < 0.01 & p_value >= 0.001 ~ '**',
              p_value < 0.001 ~ '***'
            )) %>%
            unique()

          slope_temp <- bind_rows(slope_temp, slope_frame)

          ci_bin_temp <- predict(lm_bin_temp, interval = 'confidence')
          ci_bin_temp <- data.frame(ci = ci_bin_temp)


          ci_temp <- bind_rows(ci_temp, ci_bin_temp)

        }

        temp_dat <- cbind(t, ci_temp)
        } #Slopes

      temp_dat$bin_cut_group <- bin_cut_match$bin_group[match(temp_dat$bin_cut_id, bin_cut_match$bin_cut_id)]
      temp_dat$bin_range <- bin_cut_match$bin_range[match(temp_dat$bin_cut_id, bin_cut_match$bin_cut_id)]
      temp_dat$slope <- slope_temp$slope[match(temp_dat$bin_range, slope_temp$bin)]
      temp_dat$sig <- slope_temp$sig[match(temp_dat$bin_range, slope_temp$bin)]
      temp_dat$bin_cut_group <- paste0(temp_dat$bin_cut_group, temp_dat$slope, temp_dat$sig )
      temp_dat$bin_cut_group <- factor(temp_dat$bin_cut_group, levels = unique(temp_dat$bin_cut_group))

      {
        temp_figure_stand_alone <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)', group = bin_cut_id), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = ci.lwr, ymax = ci.upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          facet_wrap(~bin_cut_group) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = var,
            x = paste("\nSteps\n(", expression("\U00B1 2 \U03C3"), ")"),
            y = 'Accuracy\n',
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
          theme(
            axis.text = element_text(size = 14),
            axis.title = element_text(size = 16, face = 'bold'),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            panel.grid = element_blank(),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "bottom",
            strip.text = element_text(size = 14, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))


      } #Stand Alone

      {
        temp_figure_combined <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)', group = bin_cut_id), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = ci.lwr, ymax = ci.upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          facet_wrap(~bin_cut_group) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = 'Rolling Bins',
            x = "\n",
            y = '\n',
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
          theme(
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 10, face = 'bold'),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            panel.grid = element_blank(),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "none",
            strip.text = element_text(size = 8, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))

      } #Combined


      diagnostic_list$figures$rolling[[var]] <- temp_figure_stand_alone
      combined_diagnostic_list$figures$rolling[[var]] <- temp_figure_combined
      diagnostic_list$slope_tables$rolling[[var]] <- slope_temp

    }

  } # Rolling Bins

  {

    data <- pai_object$push

    for (var in variables){

      temp_dat <- data.frame(data[[var]])
      temp_dat <- temp_dat[,c(1,3)]
      names(temp_dat) <- c('steps', 'accuracy')

      temp_dat$steps <- as.numeric(temp_dat$steps)
      num_bins = bins
      temp_dat$bin <- cut_interval(temp_dat$steps, n = num_bins)

      temp_dat <- temp_dat %>%
        rowwise() %>%
        mutate(cut = ifelse(is.numeric(as.numeric(gsub(".*\\,", "", gsub("\\]", "", bin)))),
                            gsub(".*\\,", "", gsub("\\]", "", bin)),
                            0)) %>%
        mutate(cut = as.numeric(cut)) %>%
        group_by(cut) %>%
        mutate(bin_id = cur_group_id())


      {
        t <- data.frame()
        bin_cut_id <- 1
        for (i in unique(temp_dat$bin_id)){
          max_bin <- max(temp_dat$bin_id)
          bin_1 <- i
          bin_2 = ifelse(bin_1 + bin_cut <= max_bin, bin_1 + bin_cut, max_bin)

          bins_collect <- c(bin_1:bin_2)

          temp_bin_dat <- temp_dat %>%
            filter(bin_id %in% bins_collect) %>%
            mutate(bin_cut_id = bin_cut_id)

          t <- bind_rows(t, temp_bin_dat)

          bin_cut_id <- bin_cut_id + 1

        }

        bin_cut_match <- t %>%
          select(steps, bin_cut_id) %>%
          group_by(bin_cut_id) %>%
          mutate(bin_range = paste0("(", round(min(steps), 3), ", ", round(max(steps), 3), "]")) %>%
          mutate(bin_group = paste0(bin_range, ' Bin (', bin_cut_id, ') \n Slope = '))

        ci_temp <- data.frame()
        slope_temp <- data.frame()
        unique_bins <- unique(t$bin_cut_id)

        for (b in unique_bins){

          temp_bin <- t[t$bin_cut_id == b,]
          lm_bin_temp <- lm(accuracy ~ steps, data = temp_bin)

          summary_lm <- summary(lm_bin_temp)
          slope <- coef(summary_lm)[2]
          se_slope <- summary_lm$coefficients[2, "Std. Error"]
          p_value <- summary_lm$coefficients[2, "Pr(>|t|)"]

          slope_frame <- data.frame(
            bin = bin_cut_match$bin_range[match(b, bin_cut_match$bin_cut_id)],
            slope = round(slope, 3),
            se = round(se_slope, 3),
            p_value = round(p_value, 5)) %>%
            mutate(sig = case_when(
              .default = "",
              p_value < 0.05 & p_value >= 0.01 ~ "*",
              p_value < 0.01 & p_value >= 0.001 ~ '**',
              p_value < 0.001 ~ '***'
            )) %>%
            unique()

          slope_temp <- bind_rows(slope_temp, slope_frame)

          ci_bin_temp <- predict(lm_bin_temp, interval = 'confidence')
          ci_bin_temp <- data.frame(ci = ci_bin_temp)


          ci_temp <- bind_rows(ci_temp, ci_bin_temp)

        }

        temp_dat <- cbind(t, ci_temp)
        } #Slopes

      temp_dat$bin_cut_group <- bin_cut_match$bin_group[match(temp_dat$bin_cut_id, bin_cut_match$bin_cut_id)]
      temp_dat$bin_range <- bin_cut_match$bin_range[match(temp_dat$bin_cut_id, bin_cut_match$bin_cut_id)]
      temp_dat$slope <- slope_temp$slope[match(temp_dat$bin_range, slope_temp$bin)]
      temp_dat$sig <- slope_temp$sig[match(temp_dat$bin_range, slope_temp$bin)]
      temp_dat$bin_cut_group <- paste0(temp_dat$bin_cut_group, temp_dat$slope, temp_dat$sig )
      temp_dat$bin_cut_group <- factor(temp_dat$bin_cut_group, levels = unique(temp_dat$bin_cut_group))

      {
        temp_figure_stand_alone <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)', group = bin_cut_id), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = ci.lwr, ymax = ci.upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          facet_wrap(~bin_cut_group) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = var,
            x = paste("\nSteps\n(", expression("\U00B1 2 \U03C3"), ")"),
            y = 'Accuracy\n',
            color = 'Fit Type',
            linetype = 'Fit Type'
          ) +
          theme(
            axis.text = element_text(size = 14),
            axis.title = element_text(size = 16, face = 'bold'),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            panel.grid = element_blank(),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "bottom",
            strip.text = element_text(size = 14, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))


      } #Stand Alone

      {
        temp_figure_combined <- ggplot(temp_dat, aes(x = steps, y = accuracy)) +
          geom_point(colour = 'gray5') +
          geom_vline(aes(xintercept = cut), linetype = 2, alpha = 1/3) +
          stat_smooth(aes(colour = 'Linear Fit\n(w/ 95% CI)', group = bin_cut_id), data = temp_dat, method = "lm", se = FALSE, linetype = 'solid', size = 1) +
          geom_errorbar(aes(ymin = ci.lwr, ymax = ci.upr), width = 0, colour = 'gray5') +
          stat_smooth(aes(colour = 'Loess Fit\n(w/ SE)'), method = "loess", se = TRUE, linetype = 'dashed', size = 1) +
          facet_wrap(~bin_cut_group) +
          theme_minimal() +
          scale_color_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'gray5', 'Loess Fit\n(w/ SE)' = 'red')) +
          scale_linetype_manual(values = c('Linear Fit\n(w/ 95% CI)' = 'solid', 'Loess Fit\n(w/ SE)' = 'dashed')) +
          geom_vline(xintercept = 0, linetype = 2, colour = 'gray5', alpha = 1/4) +
          labs(
            title = 'Rolling Bins (Extended)',
            color = 'Fit Type',
            linetype = 'Fit Type',
            x = "\n",
            y = "\n"
          ) +
          theme(
            axis.text = element_text(size = 10),
            axis.title = element_text(size = 10, face = 'bold'),
            panel.border = element_rect(linewidth = 1, color = "gray5", fill = NA),
            panel.grid = element_blank(),
            legend.title.align = 0.5,
            legend.text.align = 0.25,
            legend.title = element_blank(),
            legend.text = element_text(size = 15, color = "gray5"),
            legend.box.background = element_rect(size = 1, color = 'gray5', fill = NA),
            legend.position = "none",
            strip.text = element_text(size = 8, face = "bold"),
            strip.background = element_rect(fill = "gray", color = "gray5"),
            plot.title = element_text(size = 18, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 15),
            plot.caption = element_text(size = 12, hjust = 0, face = 'italic'))

        temp_figure_combined
      } #Combined


      diagnostic_list$figures$rolling_extended[[var]] <- temp_figure_stand_alone
      combined_diagnostic_list$figures$rolling_extended[[var]] <- temp_figure_combined
      diagnostic_list$slope_tables$rolling_extended[[var]] <- slope_temp

    }

  } #Rolling Extended Bins

  {

    diagnostic_list$combined_diagnostics <- list()

    for (var in variables){

      arranged_temp <- suppressMessages(cowplot::ggdraw() +
                                          draw_plot(combined_diagnostic_list$figures$rolling[[var]], 0, 0, 0.5, 0.5) +
                                          draw_plot(combined_diagnostic_list$figures$rolling_extended[[var]], 0.5, 0, 0.5, 0.5) +
                                          draw_plot(combined_diagnostic_list$figures$linear[[var]], 0, 0.5, 0.5, 0.5) +
                                          draw_plot(combined_diagnostic_list$figures$static[[var]], 0.5, 0.5, 0.5, 0.5))

      diagnostic_list$combined_diagnostics[[var]] <- arranged_temp
    }


  } #Combined Diagnostic


  return(diagnostic_list)


}

#Types:
# static = by bin
# rolling = by bin + 1
# rolling_extended = by_bin + bin_cut:max_bin
# placebo = placebo iterations


c <- pai_diagnostic(pai_object = test,
                    bins = 6,
                    variables = NULL,
                    bin_cut = 3)

c$combined_diagnostics$var3
c$slope_tables$rolling_extended$var3

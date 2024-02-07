pai_main <- function(data,
                     outcome = NULL,
                     predictors = NULL,
                     factors = NULL,
                     assign_factors = c(FALSE, 4),
                     interactions = NULL,
                     drop_vars = NULL,
                     ml = c(NA, NA, 10, 10),
                     custom_tc = FALSE,
                     seed = 1234){

  start_time <- Sys.time()

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
      parameters[['predictors']] <- c(names(data)[!names(data) %in% parameters[['outcome']]])
    } else {
      parameters[['predictors']] <- predictors
    } #predictors

    if(is.null(factors)){
      parameters['factors'] <- 'None'
    } else {
      parameters['factors'] <- factors
    } #setting factor terms

    if (assign_factors[1] == FALSE) {
      parameters['assign_factors'] <- 'FALSE'
      parameters['factor_cutoff'] <- NULL
    } else {
      parameters['assign_factors'] <- 'TRUE'
      parameters['factor_cutoff'] <- as.numeric(assign_factors[2])
    }

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

    if (custom_tc == FALSE){
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
      "    Predictors = ", paste(unlist(parameters$predictors), collapse = ", "), "\n",
      "    Factor Terms = ", paste(unlist(parameters$factors), collapse = ", "), "\n",
      "    Automatically Assign Factors = ", ifelse(parameters$assign_factors == 'TRUE', paste0('TRUE, Level Cut-Off = ', parameters$factor_cutoff), 'FALSE'), "\n",
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

    data_type = parameters$data_type

    dat <- data %>%
      dplyr::select(any_of(c(outcome, unlist(parameters$predictors))))

    declared_interactions <- c()
    {

      if(parameters$interactions[1] == 'None'){
        declared_interactions <- NULL
        dat <- dat
      } else {
        interaction_cols <- list()
        for (int in 1:length(parameters$interactions)){
          temp_in <- unlist(stringr::str_split(parameters$interactions[int], pattern = '\\:'))
          temp_in_cols <- data %>%
            select(any_of(temp_in)) %>%
            mutate(across(all_of(temp_in), as.numeric)) %>%
            mutate(interaction = rowSums(across(everything(), .fns = ~. * .)))
          interaction_cols[[parameters$interactions[int]]] <- temp_in_cols$interaction
        }
        interaction_cols <- data.frame(interaction_cols)
        names(interaction_cols) <- gsub('\\.', '\\:', names(interaction_cols))
        declared_interactions <- names(interaction_cols)
        dat <- cbind(dat, interaction_cols)
      }

    } #Introduce Declared Interactions (If Any)


    dcols <- which(apply(dat, 2, function(x) length(unique(x)))==1)
    dcols2 <- which(apply(dat, 2, function(x) length(levels(as.factor(x))) == 1))
    dcols <- unique(dcols, dcols2)
    if (length(dcols) >0)  dat <- dat[,-dcols]

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

    runpred <- function(mod, var, stepper, Z){

      if (data_type == 'Continuous'){
        Z[[var]] <- Z[[var]] + stepper
        pred <- predict(mod, Z)
        true <- Z$y
        dif <- pred - true
        return(dif)
      } else {
        Z[[var]] <- Z[[var]] + stepper
        pred <- predict(mod, Z)
        true <- Z$y
        onecount <- length(which(pred=='1'))/length(true)
        acc <- length(which(pred==true))/length(true)
        return(c(onecount, acc))
      }



    } #runpred

    push <- function(l, combined_variables_list){

      message("    Beginning Push Protocol...")

      push_output <- list()

      factor_vars <- combined_variables_list[grepl('as.factor', combined_variables_list)]
      factor_vars <- gsub('as\\.factor\\(', '', gsub('\\)', '', factor_vars))
      all_vars <- gsub('as\\.factor\\(', '', gsub('\\)', '', combined_variables_list))

      for (variable in 1:length(names(l$with.test))){
        x = names(l$with.test)[variable]
        Z = l$with.test

        if (x %in% factor_vars || x == 'y' || is.factor(Z[,x])){
          next
        }  else{
          sdx <- sd(Z[,x])
          steps <- seq(-2*sdx, 2*sdx, (4*sdx)/100)
        }


        tester <- lapply(steps, function(z) runpred(l$with, x, z, Z ))
        runpred_output <- cbind(steps, t(list.cbind(tester)))
        push_output[[x]] <- runpred_output

      }

      return(push_output)


    } #push

    placebo_shuffle <- function(w, d.train, d.test, combined_variables_list) {

      message("    Beginning Placebo Protocol...")

      placebos <- data.frame() # Initialize Empty DF
      vars <- combined_variables_list
      original_predictions <- predict(w, d.test, na.action = na.pass) #Original Predictions from Mod.With

      for (rep in 1:as.numeric(parameters$placebo_iterations)) {
        for (variable in vars) {

          shuffle_data <- d.test
          shuffle_data[[variable]] <- sample(shuffle_data[[variable]]) # Shuffle the variable

          shuffled_predictions <- predict(w, newdata = shuffle_data, na.action = na.pass) # Predict using the shuffled data

          accuracy_change <- mean(original_predictions != shuffled_predictions) # Calculate accuracy change

          placebo_temp <- data.frame(rep_count = rep, variable = variable, accuracy_change = accuracy_change) # Store the accuracy change
          placebos <- bind_rows(placebos, placebo_temp)
        }

        if (rep %% 5 == 0) {
          cat(paste0('           Completed Placebo Shuffling Iteration ', rep, '\n'))
        }

      }


      return(placebos)

    }

    dropping_vars <- function(mod.with, d.test, combined_variables_list){

      message("    Beginning Variable Omission Protocol...")

      fit_change <- data.frame() #Initialize Empty DF for Fit Changes by Drop Var

      if (is.null(parameters$drop_vars)){
        drop <- combined_variables_list
      } else if (parameters$drop == 'All Predictors'){
        drop <- combined_variables_list
      } else {
        drop <- unique(parameters$drop_vars)
      } #Declare Dropvars by DropVar Type

      generate_combinations_df <- function(drop) {

        combinations <- list()

        for (d in 1:length(drop)){
          other_predictors <- combined_variables_list[!combined_variables_list %in% c(drop[d], paste0('as.factor(', drop[d], ')'))]
          combinations[[paste('Combination_', d)]] <- data.frame(combination = paste(other_predictors, collapse = ", "), dropped = drop[d])
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
      #full_dat = dat

      d <- full_dat[!names(full_dat) %in% names(outcome_var)]
      d <- cbind(y, d)
      names(d)[1] <- 'y'
      if (data_type == 'Binomial'){
        d$y <- as.factor(d$y)
      } #Create DF for Train/Test Split


      d.train <- d[seq(ntrain),] #Set Training Data
      d.test <- d[-seq(ntrain),] # Set Testing Data

      {

        single_level_cols <- c()

        for (i in 1:ncol(d.train)){
          temp_col <- names(d.train)[i]
          temp_levels <- length(levels(as.factor(d.train[,i])))
          if (as.numeric(temp_levels) == 1){
            single_level_cols <- c(single_level_cols, temp_col)
          }
        }

        if (length(single_level_cols) > 0){
          message("           Note -- Removing Following Vars Due to Insufficient Levels in Test/Training Data: ", paste(single_level_cols, collapse = ", "))

        }



      } #Indicate Which Vars to Remove B/C Insufficient Levels in Factor Form

      d.train <- d.train[!names(d.train) %in% single_level_cols]
      d.test <- d.test[!names(d.test) %in% single_level_cols]


      {

        variables = names(d.train)[!names(d.train) %in% 'y']
        factor_terms <- c()
        declared_factors <- c()
        assigned_factors <- c()

        {
          if (parameters$factors[1] == 'None'){
            declared_factor_terms <- NULL
          } else{

            for (i in 1:length(parameters$factors)){
              declared_factors[i] <- paste0('as.factor(', parameters$factors[i], ')')
            }

          }
        } #Introduce Declared Factors (If Any)

        {
          remaining_vars <- names(d.train)[!names(d.train) %in% 'y']
          remaining_vars <- remaining_vars[!remaining_vars %in% unique(parameters$factors)]
          remaining_vars <- remaining_vars[!grepl('\\:', remaining_vars)]

          if (parameters$assign_factors[1] == 'FALSE'){ #If Assign_Factors = FALSE - Skip
            assigned_factors <- remaining_vars
          } else {

            for (i in 1:length(remaining_vars)){
              temp_var <- remaining_vars[i] #Get Var
              temp_col <- d.train[temp_var][,1] #Subset Data by Var

              if (is.character(temp_col)){
                assigned_factors[i] <- paste0('as.factor(', temp_var, ')') #If Character - Assign to Factor
              } else if (length(unique(temp_col)) < as.numeric(parameters$factor_cutoff)) {
                assigned_factors[i] <- paste0('as.factor(', temp_var, ')') #If Unique Values < Assigned Factor Cutoff -- Assign to Factor
              } else {
                assigned_factors[i] <- temp_var #If Not Factor - Return as Simple Var
              }

            }

          }

        } #Auto Assign Factors (If TRUE)

        combined_vars <- c()

        if (!is.null(declared_factors)) {
          combined_vars <- c(combined_vars, unlist(declared_factors))
        }
        if (!is.null(assigned_factors)) {
          combined_vars <- c(combined_vars, unlist(assigned_factors))
        }
        if (!is.null(declared_interactions)) {
          interaction_terms <- c()
          for (i in 1:length(declared_interactions)){
            terms <- unlist(stringr::str_split(declared_interactions[i], pattern = "\\:"))
            factor_check <- c()
            for (t in 1:length(terms)){
              temp_term <- terms[t]
              factor_check[t] <- ifelse(paste0('as.factor(', temp_term, ')') %in% combined_vars, paste0('as.factor(', temp_term, ')'), temp_term)
            }
            interaction_terms <- c(interaction_terms, paste(factor_check, collapse = ":"))
          }

          combined_vars <- c(combined_vars, interaction_terms)
        }

        combined_vars <- unique(combined_vars)

        single_level_to_toss <- c(single_level_cols, paste0('as.factor(', single_level_cols, ')'))
        combined_vars <- combined_vars[!combined_vars %in% single_level_to_toss]

        if(data_type == 'Binomial'){
          formula = paste0('as.factor(y) ~ ', paste(combined_vars, collapse = " + "))
          formula = as.formula(formula)
        } else {
          formula = paste0('y ~ ', paste(combined_vars, collapse = " + "))
          formula = as.formula(formula)
        }


      }  #Compile Formula for ML


      message("    Compiling Baseline ", parameters$ml_model, "...")

      capture_output_mod.with <- capture.output({

        suppressWarnings(

          mod.with <- train(form = as.formula(formula),
                            data = d.train,
                            metric = ifelse(parameters$data_type == 'Binomial', 'Accuracy', 'RMSE'),
                            method = as.character(parameters$ml_model),
                            trControl = tc_main,
                            localImp = TRUE)


        )
      }) #Get mod.with

      message("    Compiling Default Random Forest...")

      tc_rf <- trainControl(method = 'repeatedcv', number = cv_folds, repeats = 3, savePredictions = TRUE) #TC for Basic RF
      capture_output_rf.basic <- capture.output({

        suppressWarnings(

          rf.basic <- train(form = formula,
                            data = d.train,
                            metric = ifelse(parameters$data_type == 'Continuous', 'RMSE', 'Accuracy'),
                            method = 'rf',
                            trControl = tc_main,
                            localImp = TRUE)

        )
      }) #Get Basic RF

      placebo_base <- placebo_shuffle(mod.with, d.train, d.test, combined_vars) #Initialize Placebo_Shuffle
      placebo <- placebo_base %>%
        select(-rep_count) %>%
        rename(var = variable) %>%
        group_by(var) %>%
        summarize(min_change = quantile(accuracy_change, 0.025),
                  max_change = quantile(accuracy_change, 0.975)) #Post Process Placebo_Shuffle

      fit_change <- dropping_vars(mod.with, d.test, combined_vars) #Initialize Dropping_Vars
      fit_assess <- left_join(fit_change, placebo, by = 'var') #Join Fit Changes by Dropped Var ('var')

      output_list <- list(
        input_parameters = parameters,
        with = mod.with,
        rf.basic = rf.basic,
        X = full_dat,
        with.test = d.test,
        with.train = d.train,
        var = combined_vars,
        drop_acc.ch = fit_assess,
        placebo = placebo) #Compile All Into Single List Object

      pusher <- push(output_list, combined_vars) #Initialize push
      output_list$push = pusher #Put push output in output_list

      return(output_list)



    } #Main Function



  } #Core Functions

  pai_output <- pai_ml(y = outcome_var, full_dat = dat, ntrain = train.set, predictors = parameters$predictors, interactions = unlist(parameters['interactions'])) #Initialize Core Functions

  end_time <- Sys.time()
  difference_time = end_time-start_time

  message("-------------------------------------------------------------------")
  cat("-------------------------- PAI  Complete --------------------------\n")
  message("-------------------------------------------------------------------")
  message('Completion Time = ', round((difference_time/60), 2), ' Minutes')

  return(pai_output)



}


test <- pai_main(data = with.dat,
                 outcome = 'direction',
                 predictors = c('term', 'issueArea', 'iA.8', 'iA.7', 'iA.2', 'iA.1'),
                 factors = NULL,
                 assign_factors = c(TRUE, 4),
                 interactions = NULL,
                 drop_vars = NULL,
                 ml = c('parRF', 8, 100, 10),
                 seed = 1234)


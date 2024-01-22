
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
                 ml = c('rf', 8, 2, 5),
                 custom_tc = "method = 'repeatedcv', number = cv_folds, repeats = 2, savePredictions = TRUE",
                 seed = 1234)

data = test_data
outcome = 'y'
predictors = c('var1', 'var2', 'var3')
interactions = 'var1*var2'
drop_vars = NULL
ml = c('rf', 8, 2, 5)
custom_tc = "method = 'repeatedcv', number = cv_folds, repeats = 2, savePredictions = TRUE"
seed = 1234


y = outcome_var
full_dat = dat
ntrain = train.set
predictors = predictors
interactions = interactions

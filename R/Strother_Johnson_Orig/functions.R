# Sandbox trial for RF idea
library(randomForest)
library(caTools)
library(e1071)
library(foreach)
library(import)
library(caret)
library(doParallel)
library(tictoc)
library(rlist)
library(randomForestExplainer)
numCores <- detectCores()

seed = 254

'%!in%' <- function(x,y)!('%in%'(x,y))


ttsplit <- function(Z, size=.2,s=seed){
  dotx <- NULL
  dotx <- c(grep('.x', names(Z)), grep('.y', names(Z)), grep('WHRehnquist', names(Z)))
  if (length(dotx)>0) Z <- Z[,-dotx]
  set.seed(s)
  rows <- dim(Z)[1]
  test.size <- floor(size*rows)

  re.rows <- sample(seq(rows))
  Z <- Z[re.rows,]
  Z.test <- Z[seq(test.size),]
  Z.train <- Z[-seq(test.size),]
  amici <- c('wlf', 'aclu', 'chamber')
  d <- NULL
  for (a in amici){
    d <- c(d, which(Z.test[[a]] %!in% Z.train[[a]]))
  }
  if(length(d)>0) Z.test = Z.test[-d,]

  olist <- return(list(test=Z.test, train=Z.train))
  return(olist)
}

repeatedforest <- function(y, Z, folds=10, r=3){
  tic()
  set.seed(seed)
  cls = makeCluster(numCores-1)
  registerDoParallel(cls)

  ydex=which(names(Z)==y)
  Z[,ydex] <- as.factor(Z[,ydex])
  dcols <- which(apply(Z,2,function(x) length(unique(x)))==1)
  if (length(dcols) >0)  Z <- Z[,-dcols]
  f <- as.formula(paste(y,'~.'))
  tc <- trainControl(method = 'repeatedcv',
                     number = folds,
                     repeats = r,
                     savePredictions = TRUE)
  rf.mod <- train(f, method = "parRF",
                  data=Z,
                  trControl = tc, importance=TRUE,
                  localImp=TRUE)
  stopCluster(cls)
  toc()
  return(rf.mod)
}

oosacc <- function(mod, Z, y){
  ydex=which(names(Z)==y)
  trues <- Z[,ydex]
  preds <- predict(mod, Z)
  return(length(which(trues==preds))/length(preds))
}

monocheck <- function(r){
  if (length(unique(r))==1) return(1) # prediction insensitive to mood
  ones <- max(which(r==1))
  twos <- min(which(r==2))
  if (ones < twos) return(2) # predictions show weak monotonicity
  if (ones > twos) return(3) # predictions do NOT show weak monotonicity

}

casemonocheck <- function(m){
  return(unlist(apply(m,1,monocheck)))
}

predforest <- function(y, x, forest, dat, stepper){
  #for (i in x){
  #  dat[,x]=(stepper)*(dat[,x]!=0)
  #}
  dat[,x][dat[,x]!=0] = stepper
  pred <- predict(forest, dat)
  true <- dat[[y]]
  onecount <- mean(as.numeric(pred)-1) #length(which(pred=='1'))/length(true)
  #acc <- length(which(pred==true))/length(true)
  olist <- list(pred=pred, ones=onecount)
  return(olist) #onecount
}

push <- function(y, forest, dat, im=TRUE){
  #sd.gm <- sd(dat$general_mood, na.rm=TRUE)
  #steps.gm <- seq(-2*sd.gm, 2*sd.gm, (4*sd.gm)/100)
  #range.gm <- max(dat$general_mood, na.rm=TRUE) - min(dat$general_mood, na.rm=TRUE)
  #steps.gm <- seq(min(dat$general_mood, na.rm=TRUE), max(dat$general_mood, na.rm=TRUE), range.gm/100)

  moodrange <- modlist$gm.only.dat$train$general_mood
  moodrange.max <- max(moodrange, na.rm=TRUE)
  moodrange.min <- min(moodrange, na.rm=TRUE)
  steps.gm <- seq(moodrange.min, moodrange.max, (moodrange.max-moodrange.min)/100)

  moodrange.im <- modlist$with.dat$train$issuemood
  moodrange.im.max <- max(moodrange.im, na.rm=TRUE)
  moodrange.im.min <- min(moodrange.im, na.rm=TRUE)
  steps.im <- seq(moodrange.im.min, moodrange.im.max, (moodrange.im.max-moodrange.im.min)/100)


  gm.names <- grep('general_mood', names(dat))
  tester.gm <- lapply(steps.gm, function(z) predforest(y, gm.names, forest, dat, z))
  tots <- fulls <- list()
  for (i in 1:length(tester.gm)) tots[[i]]=tester.gm[[i]]$ones
  for (i in 1:length(tester.gm)) fulls[[i]]=tester.gm[[i]]$pred
  #fulls <- list.cbind(fulls)
  #length(unique(m))
  #all(m==sort(m))
  #t <- cbind(steps.gm, t(list.cbind(tester.gm)))
  t <- cbind(steps.gm, unlist(tots))
  f <- list.cbind(fulls)
  f.check.gm <- casemonocheck(f) # 1 - insiensitive to mood, 2 - weak monotonicity, 3 - not monotonic
  full.preds <- list(gm=f)



  if (im==TRUE){
    #sd.im <- sd(dat$issuemood, na.rm=TRUE)
    #steps.im <- seq(-2*sd.im, 2*sd.im, (4*sd.im)/100)
    #range.im <- max(dat$issuemood, na.rm=TRUE) - min(dat$issuemood, na.rm=TRUE)
    #steps.im <- seq(min(dat$issuemood, na.rm=TRUE), max(dat$issuemood, na.rm=TRUE), range.im/100)

    im.names <- grep('issuemood', names(dat))
    tester.im <- lapply(steps.im, function(z) predforest(y, im.names, forest, dat, z))
    tots.im <- fulls.im <- list()
    for (i in 1:length(tester.im)) tots.im[[i]]=tester.im[[i]]$ones
    for (i in 1:length(tester.im)) fulls.im[[i]]=tester.im[[i]]$pred
    f.im <- list.cbind(fulls.im)
    f.check.im <- casemonocheck(f.im)
    t <- cbind(t, steps.im, unlist(tots.im))
    full.preds$im = f.im
  }

  #if (j %!in% c('AJGoldberg', 'EKagan')) {
  range.mq <- max(dat$mqmed, na.rm=TRUE) - min(dat$mqmed, na.rm=TRUE)
  steps.mq <- seq(min(dat$mqmed, na.rm=TRUE), max(dat$mqmed, na.rm=TRUE), range.mq/100)
  mq.names <- grep('mqmed', names(dat))
  tester.mq <- lapply(steps.mq, function(z) predforest(y, mq.names, forest, dat, z))
  tots.mq <- fulls.mq <- list()
  for (i in 1:length(tester.mq)) tots.mq[[i]]=tester.mq[[i]]$ones
  for (i in 1:length(tester.mq)) fulls.mq[[i]]=tester.mq[[i]]$pred
  f.mq <- list.cbind(fulls.mq)
  full.preds$mq = f.mq
  f.mq.check <- casemonocheck(f.mq)
  t <- cbind(t, steps.mq, unlist(tots.mq))
  #}

  outlist = list(t=t, f=full.preds)
  return(outlist)
}



permute.test <- function(mod, test, samps=200, s=seed){

  set.seed(254)
  moods <- grep('mood', names(test))
  sals <- unique(c(grep('salience', names(test)), grep('CLR', names(test)), grep('sal.', names(test)), grep('Sal.', names(test))))
  lc <- grep('lcD', names(test))
  jr <- grep('judRev', names(test))
  lats <- grep('lat', names(test))
  issues <- which(names(test) %in% c('issueArea', "issuecluster", "issueFactor",
                                     "iA.2", "iA.1", "iA.7", "iA.8", "adminAction", "adminActionState"))
  amicis <- which(names(test) %in% c("amaff", "amrev", 'totam', 'wlf', 'chamber', 'sg', 'aclu'))
  ideo <- which(names(test) %in% c('scmed', 'mqmed', 'mqmean'))
  groups <- list(moods=moods, sals=sals, lc=lc, jr=jr, lats=lats, issues=issues, amicis=amicis, ideo=ideo)

  n=dim(test)[2]
  r=dim(test)[1]
  namer <- names(test)
  resamps <- preds <- NULL
  base.pred <- predict(mod, test)
  l <- length(base.pred)
  base.pred <- length(which(base.pred==test$direction))/length(base.pred)
  for (i in 1:samps){
    resamps <- cbind(resamps, sample(seq(r)))
  }
  namelist <- NULL
  preds <- NULL
  #for (j in 1:n){
  #  if (namer[j]=='direction') next
  #  print(namer[j])
  #  namelist <- c(namelist, namer[j])
  for (g in groups){
    #p.col <- base.pred
    #for (k in 1:samps){
    #p2 <-  apply(resamps, 2, function(k) {
    p2 <- NULL
    for (i in 1:samps){
      df <- test
      for (j in g){
        df[,j] <- df[resamps[,i],j]
      }
      #df[,g] <- df[resamps[,i],g]
      pred <- predict(mod, df)
      p <- length(which(pred==df$direction))/length(base.pred)
      p2 <- c(p2, p)
      #return(p)
    }
    #)
    preds <- cbind(preds, p2)
  }
  pred.df <- as.data.frame(preds)
  names(pred.df) <- names(groups) #namelist
  olist <- list(preds = pred.df, base = base.pred, pred.num=l,
                means = apply(pred.df, 2, mean)/l,
                mins = apply(pred.df, 2, min)/l,
                upper = apply(pred.df, 2, function(x) quantile(x, .95))/l)
  return(olist)
}




## function to run/save random forest models for justice

RF.run <- function(j){
  print(j)
  #load(paste0(j,'RF.RData'))

  #if (j!='Cases') next

  # Pull dataframes
  with.dat <- ttsplit(perms[[j]][[1]])
  without.dat <- ttsplit(perms[[j]][[2]])
  gm.only.dat <- ttsplit(perms[[j]][[3]])


  f = 10
  reps = 5

  # Run the Random Forest model on the real data
  RF.with <- repeatedforest('direction', with.dat$train, folds=f, r=reps)
  RF.without <- repeatedforest('direction', without.dat$train, folds=f, r=reps)
  RF.gm.only <- repeatedforest('direction', gm.only.dat$train, folds=f, r=reps)

  modlist <- list(RF.with= RF.with, RF.without=RF.without, RF.gm.only=RF.gm.only,
                  with.dat=with.dat, without.dat=without.dat, gm.only.dat=gm.only.dat)


  if (j=='Cases') {
    no.issues.dat <- ttsplit(perms[[j]][[4]])
    no.amici.dat <- ttsplit(perms[[j]][[5]])
    no.ideo.dat <- ttsplit(perms[[j]][[6]])
    no.lc.dat <- ttsplit(perms[[j]][[7]])
    no.jr.dat <- ttsplit(perms[[j]][[8]])
    no.lats.dat <- ttsplit(perms[[j]][[9]])
    no.issues.dat.gmonly <- ttsplit(perms[[j]][[10]])
    no.amici.dat.gmonly <- ttsplit(perms[[j]][[11]])
    no.ideo.dat.gmonly <- ttsplit(perms[[j]][[12]])
    no.lc.dat.gmonly <- ttsplit(perms[[j]][[13]])
    no.jr.dat.gmonly <- ttsplit(perms[[j]][[14]])
    no.lats.dat.gmonly <- ttsplit(perms[[j]][[15]])

    no.sals.dat <- ttsplit(perms[[j]][[16]])
    no.sals.dat.gmonly <- ttsplit(perms[[j]][[17]])


    modlist$no.issues.dat <- no.issues.dat
    modlist$no.amici.dat <- no.amici.dat
    modlist$no.ideo.dat <- no.ideo.dat
    modlist$no.lc.dat <- no.lc.dat
    modlist$no.jr.dat <- no.jr.dat
    modlist$no.lats.dat <- no.lats.dat
    modlist$no.issues.dat.gmonly <- no.issues.dat.gmonly
    modlist$no.amici.dat.gmonly <- no.amici.dat.gmonly
    modlist$no.ideo.dat.gmonly <- no.ideo.dat.gmonly
    modlist$no.lc.dat.gmonly <- no.lc.dat.gmonly
    modlist$no.jr.dat.gmonly <- no.jr.dat.gmonly
    modlist$no.lats.dat.gmonly <- no.lats.dat.gmonly

    modlist$no.sals.dat <- no.sals.dat
    modlist$no.sals.dat.gmonly <- no.sals.dat.gmonly

    RF.issues <- repeatedforest('direction', no.issues.dat$train, folds=f, r=reps)
    RF.amici <- repeatedforest('direction', no.amici.dat$train, folds=f, r=reps)
    RF.ideo <- repeatedforest('direction', no.ideo.dat$train, folds=f, r=reps)
    RF.lc <- repeatedforest('direction', no.lc.dat$train, folds=f, r=reps)
    RF.jr <- repeatedforest('direction', no.jr.dat$train, folds=f, r=reps)
    RF.lats <- repeatedforest('direction', no.lats.dat$train, folds=f, r=reps)
    RF.sals <- repeatedforest('direction', no.sals.dat$train, folds=f, r=reps)

    RF.issues.gmonly <- repeatedforest('direction', no.issues.dat.gmonly$train, folds=f, r=reps)
    RF.amici.gmonly <- repeatedforest('direction', no.amici.dat.gmonly$train, folds=f, r=reps)
    RF.ideo.gmonly <- repeatedforest('direction', no.ideo.dat.gmonly$train, folds=f, r=reps)
    RF.lc.gmonly <- repeatedforest('direction', no.lc.dat.gmonly$train, folds=f, r=reps)
    RF.jr.gmonly <- repeatedforest('direction', no.jr.dat.gmonly$train, folds=f, r=reps)
    RF.lats.gmonly <- repeatedforest('direction', no.lats.dat.gmonly$train, folds=f, r=reps)
    RF.sals.gmonly <- repeatedforest('direction', no.sals.dat.gmonly$train, folds=f, r=reps)

    modlist$RF.issues <- RF.issues
    modlist$RF.amici <- RF.amici
    modlist$RF.ideo <- RF.ideo
    modlist$RF.lc <- RF.lc
    modlist$RF.jr <- RF.jr
    modlist$RF.lats <- RF.lats
    modlist$RF.sals <- RF.sals
    modlist$RF.issues.gmonly <- RF.issues.gmonly
    modlist$RF.amici.gmonly <- RF.amici.gmonly
    modlist$RF.ideo.gmonly <- RF.ideo.gmonly
    modlist$RF.lc.gmonly <- RF.lc.gmonly
    modlist$RF.jr.gmonly <- RF.jr.gmonly
    modlist$RF.lats.gmonly <- RF.lats.gmonly
    modlist$RF.sals.gmonly <- RF.sals.gmonly

  }


  save(modlist, file=paste0(j,'RF.RData'))
  return(modlist)
}

marginals.run <- function(j){
  n <- paste0(j,'RF.RData')
  load(n)
  m=NULL
  monotypes = list()
  # Get marginals on mood variables
  if ('issuemood' %in% names(modlist$with.dat$train)){
    m = push('direction', modlist$RF.with, modlist$with.dat$test, im=TRUE)
    monotypes$im = m$f
    marginals <- m$t
    marginals.gm <- marginals[,1:2]
    marginals.im <- marginals[,3:4]
    #marginals.mq <- marginals[,5:6]
  }

  if ('issuemood' %!in% names(modlist$with.dat$train)){
    marginals.gm <- NA
    marginals.im <- NA
    marginals.mq <- NA
  }

  # Get marginals on general mood in gm.only
  m.gm <- push('direction', modlist$RF.gm.only, modlist$gm.only.dat$test, im=FALSE)
  monotypes$gm.only = m.gm$f
  marginals <- m.gm$t
  marginals.gm.only <- marginals[,1:2]
  #marginals.gm.only.mq <- marginals[,5:6]

  # Check if in sample accuracy increases with mood data
  with.acc <- mean(modlist$RF.with$resample$Accuracy)
  without.acc <- mean(modlist$RF.without$resample$Accuracy)
  gm.only.acc <- mean(modlist$RF.gm.only$resample$Accuracy)
  improved.all <- with.acc > without.acc
  improved.gm <- gm.only.acc > without.acc

  # Check out of sample accuracy increases with mood data

  with.acc.oos <- oosacc(modlist$RF.with, modlist$with.dat$test, 'direction')
  without.acc.oos <- oosacc(modlist$RF.without, modlist$without.dat$test, 'direction')
  gm.only.acc.oos <- oosacc(modlist$RF.gm.only, modlist$gm.only.dat$test, 'direction')
  improved.all.oos <- with.acc.oos > without.acc.oos
  improved.gm.oos <- gm.only.acc.oos > without.acc.oos

  # Get accuracy of modal vote
  mod <- max(table(modlist$with.dat$test$direction)) / sum(table(modlist$with.dat$test$direction)) #out of sample

  # Row of relevant data
  outrow <- c(j, mod,
              with.acc, without.acc, gm.only.acc, improved.all, improved.gm,
              with.acc.oos, without.acc.oos, gm.only.acc.oos, improved.all.oos, improved.gm.oos)
  outlist <- list(marginals.gm=marginals.gm, marginals.im=marginals.im, #marginals.mq = marginals.mq,
                  marginals.gm.only=marginals.gm.only, #marginals.gm.only.mq=marginals.gm.only.mq,
                  outrow=outrow,
                  X.with=modlist$with.dat, X.without=modlist$without.dat, X.gm.only=modlist$gm.only.dat,
                  monotypes = monotypes,  f.im=m$f, f.gm=m.gm$f)

  Chapter3.outputs[[j]]=outlist
  save(Chapter3.outputs, file='Chapter3.RData')
  return(outlist)
}

sensitivity.run <- function(j, modlist=modlist){
  n <- paste0(j,'RF.RData')
  load(n)

  RF.with <- modlist$RF.with
  RF.without <- modlist$RF.without
  RF.gm.only <- modlist$RF.gm.only

  with.dat <- modlist$with.dat$test
  without.dat <- modlist$without.dat$test
  gm.only.dat <- modlist$gm.only.dat$test

  with.sensitivity <- permute.test(RF.with, with.dat)
  without.sensitivity <- permute.test(RF.without, without.dat)
  gm.only.sensitivity <- permute.test(RF.gm.only, gm.only.dat)

  olist <- list(with=with.sensitivity, without=without.sensitivity, gm.only=gm.only.sensitivity) #
  return(olist)
}

oos.predict <- function(dat, mod, y='direction'){
  p <- predict(mod, dat$test)
  l <- length(which(p==dat$test[[y]]))
  return(l/length(p))
}

runjustice <- function(j){
  tic()
  modlist <- RF.run(j)
  marg <- marginals.run(j)
  sense <- sensitivity.run(j)
  info <- list(modlist=modlist, sense=sense, marg=marg)
  save(info, file=paste0(j,'-FullData.RData'))
  toc()
}


############################################################################
############################################################################
############################################################################
######.    Older functions that aren't used in wrapper
############################################################################
############################################################################
############################################################################


old.permute.test <- function(y, datlist, p.start=3, p.end=12){
  olist <- list()
  for (i in seq(p.start, p.end)){
    dat = datlist[[p]]
    olist[[i]] = repeatedforest(y, dat)
  }
  return(olist)
}

subset.wrapper <- function(n){
  fname <- paste0(n,'RF.RData')
  load(fname)
  return(subset.wrapper.inner(info$modlist))
}

subset.wrapper.inner <- function(l){
  both <- l$RF.with
  gm.only <- l$RF.gm.only
  none <- l$RF.without

  both.jr <- subset.analysis.jr(both)
  gm.only.jr <- subset.analysis.jr(gm.only)
  none.jr <- subset.analysis.jr(none)

  pct.imp.both.jr <- both.jr$judRev.acc - none.jr$judRev.acc
  pct.imp.gm.jr <- gm.only.jr$judRev.acc - none.jr$judRev.acc

  count.imp.both.jr <- pct.imp.both.jr * none.jr$case.count.jr
  count.imp.gm.jr <- pct.imp.gm.jr * none.jr$case.count.jr

  pct.imp.both.sl <- both.jr$salLat.acc - none.jr$salLat.acc
  pct.imp.gm.sl <- gm.only.jr$salLat.acc - none.jr$salLat.acc

  count.imp.both.sl <- pct.imp.both.sl * none.jr$case.count.sl
  count.imp.gm.sl <- pct.imp.gm.sl * none.jr$case.count.sl

  both.jr.marg <- gm.jr.marg <- both.sl.marg <- gm.sl.marg <- NA
  if (length(both.jr$judRev.dex)>10){
    both.jr.marg <- sub.marginal.fun(mod=both, dex=both.jr$judRev.dex)
    gm.jr.marg <- sub.marginal.fun(mod=gm.only, dex=gm.only.jr$judRev.dex, im=FALSE)
    if (!is.na(both.jr$sl.dex)){
      both.sl.marg <- sub.marginal.fun(mod=both, dex=both.jr$sl.dex)
      gm.sl.marg <- sub.marginal.fun(mod=gm.only, dex=gm.only.jr$sl.dex, im=FALSE)
    }
  }

  olist <- list(mainlist=list(pct.imp.both.jr=pct.imp.both.jr, pct.imp.gm.jr=pct.imp.gm.jr,
                              count.imp.both.jr=count.imp.both.jr, count.imp.gm.jr=count.imp.gm.jr,
                              pct.imp.both.sl=pct.imp.both.sl, pct.imp.gm.sl=pct.imp.gm.sl,
                              count.imp.both.sl=count.imp.both.sl, count.imp.gm.sl=count.imp.gm.sl),
                both.jr.marg=both.jr.marg, gm.jr.marg=gm.jr.marg,
                both.sl.marg=both.sl.marg, gm.sl.marg=gm.sl.marg,
                case.count.none.jr=none.jr$case.count.jr, case.count.none.sl=none.jr$case.count.sl,
                case.count.both.jr=both.jr$case.count.jr, case.count.both.sl=both.jr$case.count.sl,
                case.count.gm.jr=gm.only.jr$case.count.jr, case.count.gm.sl=gm.only.jr$case.count.sl)

  return(olist) # once you rerun, you'll need to add stuff to get sal.lat in here. Create an olist.sl, and then create
  # an olist that combines them.
}

subset.analysis.jr <- function(mod){
  df <- mod$trainingData
  judRev.dex <- which(df$judRev==1)
  best.preds <- mod$pred[which(mod$pred$mtry==mod$bestTune[[1]]),]
  judRev.preds <- best.preds[best.preds$rowIndex %in% judRev.dex,]
  judRev.acc <- length(which(judRev.preds$pred==judRev.preds$obs))/dim(judRev.preds)[1]
  r.jr <- max(table(judRev.preds$rowIndex)) # this is the number of repeats
  case.count.jr <- dim(judRev.preds)[1]/r.jr

  salLat.acc=NA
  case.count.sl=NA
  sl.dex=NA

  if('lats' %in% names(df) & 'salience_term' %in% names(df)){
    sl.dex <- which(df$sal.lat==1) #which(df$lats * df$salience_term==1)
    salLat.preds <- best.preds[best.preds$rowIndex %in% sl.dex,]
    salLat.acc <- length(which(salLat.preds$pred==salLat.preds$obs))/dim(salLat.preds)[1]
    r.sl <- max(table(salLat.preds$rowIndex))
    case.count.sl <- dim(salLat.preds)[1]/r.sl
  }

  return(list(judRev.acc=judRev.acc, case.count.jr=case.count.jr,
              salLat.acc=salLat.acc, case.count.sl=case.count.sl,
              judRev.dex=judRev.dex, sl.dex=sl.dex))
}

sub.marginal.fun <- function(y='.outcome', mod, dex, im=TRUE){
  df <- mod$trainingData[dex,]
  return(push(y,mod,df,im)$t)
}

permute.test.inner <- function(g, mod=mod, test=test, r=r){
  return(apply(resamps, 2, function(k) {
    df <- test
    df[,g] <- df[k,g]
    pred <- predict(mod, df)
    p <- length(which(pred==df$direction))/r
  }
  )
  )
}

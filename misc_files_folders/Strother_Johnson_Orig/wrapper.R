#setwd("~/Dropbox/Johnson_Strother_SCOTUS ML/code")
setwd("C:/Users/bbj14/Dropbox/Johnson_Strother_SCOTUS ML/code")
#source('runForest.R')
source('functions.R')
'%!in%' <- function(x,y)!('%in%'(x,y))



#source('permutationTest.R') # generate the file 'permutation-dfs.RData'
setwd("C:/Users/bbj14/Dropbox/Johnson_Strother_SCOTUS ML/data/Chapter3-Models-New")
#setwd("~/Dropbox/Johnson_Strother_SCOTUS ML/data/Chapter3-Models-New")
load('permutation-dfs.RData') # loads the perms object

#Chapter3.outputs <- list()
load('Chapter3.RData')

load(file.path(dropbox, 'Chapter3.Rdata'))

for(j in names(perms)){
  #if (j=='Cases') next
  runjustice(j)
}



###############################


for(j in names(perms)[-seq(15)]){ #
  print(j)
  #load(paste0(j,'RF.RData'))

  if (j!='Cases') next

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
save(modlist, file=paste0(j,'RF.RData'))

# Get marginals on mood variables

if ('issuemood' %in% names(with.dat$train)){
  marginals <- push('direction', RF.with, with.dat$train, im=TRUE)
  marginals.gm <- marginals[,1:2]
  marginals.im <- marginals[,3:4]
}

if ('issuemood' %!in% names(with.dat$train)){
  marginals.gm <- NA
  marginals.im <- NA
}

# Get marginals on general mood in gm.only
marginals <- push('direction', RF.gm.only, gm.only.dat$train, im=FALSE)
marginals.gm.only <- marginals[,1:2]
#marginals.im <- marginals[,3:4]

# Check if in sample accuracy increases with mood data
with.acc <- mean(RF.with$resample$Accuracy)
without.acc <- mean(RF.without$resample$Accuracy)
gm.only.acc <- mean(RF.gm.only$resample$Accuracy)
improved.all <- with.acc > without.acc
improved.gm <- gm.only.acc > without.acc

# Check out of sample accuracy increases with mood data

with.acc.oos <- oosacc(RF.with, with.dat$test, 'direction')
without.acc.oos <- oosacc(RF.without, without.dat$test, 'direction')
gm.only.acc.oos <- oosacc(RF.gm.only, gm.only.dat$test, 'direction')
improved.all.oos <- with.acc.oos > without.acc.oos
improved.gm.oos <- gm.only.acc.oos > without.acc.oos

# Get accuracy of modal vote
m <- max(table(with.dat$test$direction)) / sum(table(with.dat$test$direction)) #out of sample

# Row of relevant data
outrow <- c(j, m,
            with.acc, without.acc, gm.only.acc, improved.all, improved.gm,
            with.acc.oos, without.acc.oos, gm.only.acc.oos, improved.all.oos, improved.gm.oos)
outlist <- list(marginals.gm=marginals.gm, marginals.im=marginals.im,
                marginals.gm.only=marginals.gm.only,
                outrow=outrow, X.with=with.dat, X.without=without.dat, X.gm.only=gm.only.dat)


Chapter3.outputs[[j]]=outlist
save(Chapter3.outputs, file='Chapter3.RData')
}


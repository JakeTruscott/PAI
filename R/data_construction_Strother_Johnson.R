################################################################################
# Creating PAI Data
# Paper Title
# Johnson, Strother, Truscott
################################################################################

################################################################################
# Load Packages
################################################################################
library(dplyr); library(mfx); library(mice); library(miceadds); library(plotrix); library(interflex); library(foreign); library(stringr)

################################################################################
# Additional Functions
################################################################################
'%!in%' <- function(x,y)!('%in%'(x,y))

################################################################################
# Load Necessary Data
################################################################################

scdb.j <- read.csv(file.path(dropbox, 'SCDB_2018_01_justiceCentered_Docket.csv'))
EM <- read.csv(file.path(dropbox, 'EpsteinMartin.csv'))
scdb.c <- read.csv(file.path(dropbox, 'SCDB_2018_01_caseCentered_Docket.csv'))
stim <- read.csv(file.path(dropbox, 'US-Public-Policy_Mood2019-19.3_proportion.csv'))
sally <- read.csv(file.path(dropbox, 'CLR_salienceAll.csv'))
genmood.y <- read.csv(file.path(dropbox, 'stimson_gen_mood.csv'))
votes <- read.csv(file.path(dropbox, 'votematrix.csv'))
codes <- read.csv(file.path(dropbox, 'PAP_issue_coding_SCOTUS.csv'))
macrolib <- read.csv(file.path(dropbox, 'macrolib.csv'))
demshare <- read.csv(file.path(dropbox, 'demshare.csv'))
mq <- read.csv(file.path(dropbox, 'justices.csv'))
load(file.path(dropbox, 'measures.Rdata'))
justices <- read.csv(file.path(dropbox, 'justicesdata.csv'))
issuemoods <- read.csv(file.path(dropbox, 'merge_Moods.csv'))
load(file.path(dropbox, 'scdbSalience.rdata'))
friends <- read.csv(file.path(dropbox, 'amici.csv'))
briefs <- read.dta(file.path(dropbox, 'briefs_14Oct2015_0630.dta'))
words <- read.csv(file.path(dropbox, 'wordcount.csv'))
lowercourt <- read.csv(file.path(dropbox, 'CoA_Shepherds_Solberg.csv'))

################################################################################
#Data Amendments
################################################################################
scdb.c$docket <- unlist(lapply(scdb.c$docket, as.character))
codes$docket <- unlist(lapply(codes$docket, as.character))

codes$docket <- stringr::str_replace_all(codes$docket, "\\s+", "")
scdb.c$docket <- stringr::str_replace_all(scdb.c$docket, '\\s+', '')
scdb.j$docket <- stringr::str_replace_all(scdb.j$docket, '\\s+', '')

codes <- unique(codes)

names(genmood.y)[1]='term'

stim <- stim[!is.na(stim$percent),]
sally <- sally[-which(is.na(sally$CLR_early)),]

names(demshare)[1] ='term'
names(demshare)[2] ='demvotes'
names(demshare)[3] ='demseats'

revs <- c(3,4,5)

EM.short <- EM %>%
  dplyr::select(caseIssuesId, moodQuarterly, moodQEcon, moodQJP, moodQFed, moodQCivLib, civlib, econ, judpower, federalism, conlaw)
EM.short[EM.short==0] <- NA

################################################################################
#data1
################################################################################

data1 <- merge(scdb.j, votes, by='caseIssuesId')
data1$WHRehnquistChief <- data1$WHRehnquistAssociate <- 999 # Split Rehnquist into associate years and chief years
data1$WHRehnquistChief[data1$chief == 'Rehnquist'] <- data1$WHRehnquist[data1$chief == 'Rehnquist']
data1$WHRehnquistAssociate[data1$chief != 'Rehnquist'] <- data1$WHRehnquist[data1$chief != 'Rehnquist']
data1 <- merge(scdbSalience, data1, all.y = T)
data1$nyt_dummy <- -(data1$nytSalience-1)
data1$cq_dummy <- -(data1$cqSalience-1)

sally2 <- sally[,c(1,54:66)]
names(sally2) <- c('caseId', 'earlyCoverage_LAT', 'argumentCoverage_LAT', 'pendingCoverage_LAT', 'decisionCoverage_LAT',
                   'earlyCoverage_WAPO', 'argumentCoverage_WAPO', 'pendingCoverage_WAPO', 'decisionCoverage_WAPO',
                   'earlyCoverage_NYT', 'argumentCoverage_NYT', 'pendingCoverage_NYT', 'decisionCoverage_NYT',
                   'CLR_early')

################################################################################
#data2
################################################################################
data2 <- merge(sally2, data1, all.y=T)


################################################################################
#data3
################################################################################
data3 <- merge(data2, genmood.y, all.x=T)
data3.1 <- merge(data3, demshare, all.x=T)
data3.5 <- merge(data3.1, macrolib, all.x=T)
data3.6 <- merge(data3.5, issuemoods, all.x = T)

################################################################################
#data4
################################################################################

data4 <- data3.6 %>%
  mutate(chief = as.character(chief),
         naturalCourt = as.character(naturalCourt)) %>%
  filter(!issueArea == 3) %>%
  filter(!is.na(term))

data4 <- data4 %>%
  merge(EM.short, all.x = T) %>%
  mutate(jname = as.character(justiceName)) %>%
  merge(justices, by = 'jname') %>%
  mutate(issueArea = case_when(
    .default = issueArea, # Shift some Due Process cases into crim
    issue == 40040 ~ 1, # Due Process, prisoner and defendant rights
    issue == 90020 ~ 1, # comity, crim pro
    issue == 90040 ~ 1, # comity, habeas
    issue == 90010 ~ 2, # comity, civil rights
    issue == 90070 ~ 2)) %>% # comity, privacy
  mutate(issuecluster = ifelse(issueArea %in% c(3, 5), 2, issueArea)) %>%
  mutate(issuemood = case_when(
    is.na(issueArea) ~ NA_real_,
    !(issueArea %in% c(1, 2, 3, 5, 7, 8)) ~ NA_real_,
    issueArea == 1 ~ LawCrime_mood,
    issueArea == 8 ~ Macroecon_mood,
    issueArea %in% c(2, 3, 5) ~ CivRights_mood,
    issueArea == 7 ~ Labor_mood)) %>%
  mutate(decisionDirection = decisionDirection -1)

data4$decisionDirection <- data4$decisionDirection - 1 # Make decisionDirection a 0/1 variable
firstj <- which(names(data4)=='HHBurton')
lastj <-which(names(data4)=='WHRehnquistChief')
data4[,seq(firstj,lastj)] <- data4[,seq(firstj,lastj)]-1
data4[,seq(firstj,lastj)][data4[,seq(firstj,lastj)]==998] =999

############
# Special addition. In 'personnel replacement' I created a list of cases that flipped because of new justice
# This will add a column for flipped votes
#data4$flipped <- data4$decisionDirection
#data4$flipped[data4$caseIssuesId %in% flippedcases] <- (data4$flipped[data4$caseIssuesId %in% flippedcases] - 1)^2

# Now we need to make salience measures
# The variable salience_term is a dummy variable that is a 1 for cases that are in the top 15% of salience
# in a given term.

# Choose which salience metric you want to use. We used withConferenceSalienceSum for original submission

############

data4 <- data4 %>%
  mutate(preConferenceSalienceSum = earlyCoverage_NYT + earlyCoverage_LAT + earlyCoverage_WAPO + argumentCoverage_NYT + argumentCoverage_LAT + argumentCoverage_WAPO,
         withConferenceSalienceSum = earlyCoverage_NYT + earlyCoverage_LAT + earlyCoverage_WAPO + argumentCoverage_NYT + argumentCoverage_LAT + argumentCoverage_WAPO + pendingCoverage_NYT + pendingCoverage_WAPO + pendingCoverage_LAT,
         highsalience_term <- salience_term <-
           midsalience_term <- nonsalience_term <- salience_sum <-
           preConference_salience_term <- early_salience_term <-
           preConference_midsalience_term <- early_midsalience_term <-
           preConference_nonsalience_term <- early_nonsalience_term <- NULL) %>%
  mutate(salience_sum = withConferenceSalienceSum,
         first_two = preConferenceSalienceSum,
         earlyOnly <- earlyCoverage_NYT + earlyCoverage_LAT + earlyCoverage_WAPO,
         mqmed <- scmed <- libCourt <- consCourt <- NULL,
         CLR_low <- CLR_high <- CLR_mid <- NULL) %>%
  mutate(CLR_85 = quantile(CLR_early, .85, na.rm = T),
         CLR_15 = quantile(CLR_early, .15, na.rm = T)) %>%
  mutate(CLR_low = ifelse(CLR_early <= CLR_15, 1, 0),
         CLR_high = ifelse(CLR_early >= CLR_85, 1, 0),
         CLR_mid = 1 - (CLR_high - CLR_low))

termlevel <- NULL
terms <- sort(unique(data4$term)) #Unique Terms

for (y in terms){
  message(y)
  temp <- data4[data4$term==y,]
  tmq <- mq[mq$term==y,]
  tsc <- measures[measures$term==y,]
  mqmedian <- median(tmq$post_mn, na.rm=T)
  scmedian <- median(tsc$scIdeology, na.rm=T)
  mqmn <- mean(tmq$post_mn, na.rm=T)
  scmn <- mean(tsc$scIdeology, na.rm=T)
  data4$mqmed[data4$term==y] = mqmedian
  data4$scmed[data4$term==y] = scmedian
  data4$mqmean[data4$term==y] = mqmn
  data4$scmean[data4$term==y] = scmn
  if (y %in% seq(2012,2017)){
    scmedian <- median(data4$scmed[data4$term==2011])
  }

  if (scmedian>.5){
    data4$libCourt[data4$term==y] = 1
    data4$consCourt[data4$term==y] = 0
  }
  if (scmedian<.5){
    data4$libCourt[data4$term==y] = 0
    data4$consCourt[data4$term==y] = 1
  }
  if (scmedian==.5){
    data4$libCourt[data4$term==y] = 0
    data4$consCourt[data4$term==y] = 0
  }

  thresh <- quantile(temp$salience_sum, .85, na.rm=T)
  highthresh <- quantile(temp$salience_sum, .95, na.rm=T)
  lowthresh <- quantile(temp$salience_sum, .15, na.rm=T)
  #print(paste0(y, ' ', lowthresh))
  if (is.na(thresh)) next
  if (thresh==0) {
    thresh = .1}
  data4$salience_term[data4$term==y] = data4$highsalience_term[data4$term==y] =
    data4$nonsalience_term[data4$term==y] = data4$midsalience_term[data4$term==y] = 0

  thresh2.high <- quantile(temp$first_two, .85, na.rm=T)
  thresh2.low <- quantile(temp$first_two, .15, na.rm=T)

  thresh.early.high <- quantile(temp$earlyOnly, .85, na.rm=T)
  thresh.early.low <- quantile(temp$earlyOnly, .15, na.rm=T)

  data4$salience_term[data4$term==y & data4$salience_sum >= thresh] = 1
  data4$highsalience_term[data4$term==y & data4$salience_sum >=highthresh] = 1
  data4$nonsalience_term[data4$term==y & data4$salience_sum <= lowthresh] = 1
  data4$midsalience_term[data4$term==y & data4$salience_sum > lowthresh & data4$salience_sum < thresh] = 1

  data4$preConference_salience_term[data4$term==y] = data4$preConference_nonsalience_term[data4$term==y] =
    data4$preConference_midsalience_term[data4$term==y] = data4$early_salience_term[data4$term==y] =
    data4$early_nonsalience_term[data4$term==y] = data4$early_midsalience_term[data4$term==y] = 0

  data4$preConference_salience_term[data4$term==y & data4$first_two >= thresh2.high] = 1
  data4$preConference_nonsalience_term[data4$term==y & data4$first_two <= thresh2.low] = 1
  data4$preConference_midsalience_term[data4$term==y & data4$first_two > thresh2.low & data4$first_two < thresh2.high] = 1

  data4$early_salience_term[data4$term==y & data4$earlyOnly >= thresh.early.high] = 1
  data4$early_nonsalience_term[data4$term==y & data4$earlyOnly <= thresh.early.low] = 1
  data4$early_midsalience_term[data4$term==y & data4$earlyOnly > thresh.early.low & data4$earlyOnly < thresh.early.high] = 1


} #Term-Level Meta

################################################################################
#Additional data4
# Make some variables for justice analysis
################################################################################

data4 <- data4 %>%
  mutate(logsalience = log(salience_sum + 1),
         zerosalience = ifelse(salience_sum == 0, 1, 0),
         midsalience2 = ifelse(highsalience_term != 1 & salience_term == 1, 1, 0), #Log, Zero, & Mid Salience
         USaparty = case_when(
           .default = 0,
           petitioner == 27 ~ 1,
           respondent == 27 ~ 1), # US a party
         stateaparty = case_when(
           .default = 0,
           petitioner == 28 ~ 1,
           respondent == 28 ~ 1), # State a party
         lcDispositionDirection = lcDispositionDirection  -1,
         lcDispositionDirection = ifelse(lcDispositionDirection == 2, NA, lcDispositionDirection ), # lower court disposition (make unclear ones NA)
         issueFactor = ifelse(issueArea %in% c(13, 13, 6, 12), 999, issueArea), # issueArea (collapsing Misc., Private Action, Attorneys, and Tax)
         libvotes = case_when(
           .default = NA,
           decisionDirection == 1 & !is.na(decisionDirection) ~ majVotes,
           decisionDirection == 0 & !is.na(decisionDirection) ~ minVotes),  # Add a column for number of liberal votes in a case
         data4$certConflict <- data4$certImportant <- data4$certNot <- 0,
         certConflict = ifelse(certReason %in% c(2, 3, 4, 5, 6), 1, 0),
         certImportant = ifelse(certReason %in% c(3, 10), 1, 0),
         certNot = ifelse(certReason == 1, 1, 0), # cert variables
         OurlawType = ifelse(lawType %in% c(8, 9, 5, 4), 999, lawType),
         OurlawType = ifelse(OurlawType == 2, 1, OurlawType), # Recode lawType so that all Con law together and then generate holdout category including (Other, State regulations, no legal provisions, and Court rules)
         lastyear <- nextyear <- look.back <- look.ahead <- casechange <- yr_sal_ratio <- NA) # Add columns to provide the changes in general mood

for (t in seq(min(data4$term), max(data4$term))){
  if ((t-1) %in% unique(data4$term)) {
    lastyear <- unique(data4$general_mood[data4$term==(t-1)])
  }
  if ((t-1) %!in% unique(data4$term)) {
    lastyear <- NA
  }
  if ((t+1) %in% unique(data4$term)) {
    nextyear <- unique(data4$general_mood[data4$term==(t+1)])
  }
  if ((t+1) %!in% unique(data4$term)) {
    nextyear <- NA
  }
  thisyear <- unique(data4$general_mood[data4$term==t])
  if (is.na(thisyear)) next
  data4$look.back[data4$term==t] = (thisyear - lastyear)
  data4$look.ahead[data4$term==t] = (nextyear - thisyear)
  data4$lastyear[data4$term==t] = lastyear
  data4$nextyear[data4$term==t] = nextyear
  data4$yr_sal_ratio[data4$term==t] = data4$salience_sum[data4$term==t]/mean(data4$salience_sum[data4$term==t], rm.na=T)
}

lowercourt$usVol <- sapply(lowercourt$us1, function(x) as.numeric(stringr::str_split(x, '/',simplify=T))[1])
lowercourt$usPage <- sapply(lowercourt$us1, function(x) as.numeric(stringr::str_split(x, '/',simplify=T))[2])
lowercourt <- lowercourt[which(!is.na(lowercourt$usVol)),]
friends$ledCite <- paste(friends$led1,'L. Ed. 2d', friends$led2)
data4$usVol <- sapply(data4$usCite, function(x) as.character(as.numeric(stringr::str_split(x, ' U.S. ',simplify=T)))[1])
data4$usPage <- sapply(data4$usCite, function(x) as.character(as.numeric(stringr::str_split(x, ' U.S. ',simplify=T)))[2])
data4 <- merge(data4,friends,by='ledCite', all.x=T)
data4 <- merge(data4,words,by='caseId', all.x=T)

save(data4, file = 'R/main_data4.rdata')

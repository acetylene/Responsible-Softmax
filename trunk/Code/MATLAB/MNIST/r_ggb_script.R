#!/usr/bin/env Rscript
library(ggb)
args <- commandArgs(trailingOnly=TRUE)

# test if there is at least one argument: if not, return an error
if (length(args)==0) {
  stop("At least one argument must be supplied (input file).csv", call.=FALSE)
} else if (length(args)==1) {
  # default output file
  args[2] <-"out.csv"
  args[3] <-.Machine$double.eps
}

infile <-args[1]
outfile <-args[2]
pre <- getwd()
infilename <-paste(pre,infile,sep = '/')
outfilename <-paste(pre,outfile,sep = '/')

images <- read.csv(infilename,header = FALSE,sep = ",")

g <- igraph::graph.lattice(c(20,20))
S <- cov(images)
mineig <- args[3]
fitg <- ggb(S,g,type = "global",delta = mineig)
fitg$type <- "global"
cvg <- cv_ggb(images,fitg,g,nfolds = 4)

idx <- which.min(cvg$m+exp(-cvg$lambda))
Sighat <- fitg$Sig[[idx]]
covhat <- as.matrix(Sighat)

write.table(as.matrix(covhat), file = outfilename, sep = ",", row.names = FALSE, col.names = FALSE)

# #/  Deprecated. Probably doesn't work
# fitl = ggb(S,g,type = "local",delta = mineig)
# fitl$type="local"
# cvl = cv_ggb(images,fitl,g,nfolds = 4)
# 
# #test to see which fit works best, global or local
# globalbest = cvg$m[[cvg$ibest]]+exp(-cvg$lambda_best)
# localbest = cvl$m[[cvl$ibest]]+exp(-cvl$lambda_best)
# 
# globloc = which.min(c(globalbest,localbest))
# 
# #choose the fit that produces the least cv error and  the largest lambda.
# #this guarantees a minimum locality
# if (globloc == 1){
#   Sighat=fitg$Sig[[cvg$ibest]] 
# }else{
#   Sighat=fitl$Sig[[cvl$ibest]]
# }
# #
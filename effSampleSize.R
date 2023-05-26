


library(coda)


rAR1 = function(n,sigma,alpha){
  
  eps = rnorm(n,0,sigma)
  
  x = rep(0,n)
  
  x[1] = eps[1]
  
  for (i in 2:n) {
    
    x[i] = alpha*x[i-1] + eps[i]
  }
  return(x) 
}


effectiveSize(1:1000)
effectiveSize(rnorm(1000))

effectiveSize(rAR1(1000,1,0.9))
effectiveSize(rAR1(1000,1,0.5))
effectiveSize(rAR1(1000,1,0.1))
effectiveSize(rAR1(1000,1,-0.1))
effectiveSize(rAR1(1000,1,-0.5))
effectiveSize(rAR1(1000,1,-0.9))




### scale 0.1 ################

### COV 0 

whiteCov0 = read.csv("whitecov00p1.csv",header=F)
whiteSampCov0 = apply(whiteCov0, 2, effectiveSize)

centerCov0 = read.csv("centercov00p1.csv",header=F)
centerSampCov0 = apply(centerCov0, 2, effectiveSize)

interCov0 = read.csv("intercov00p1.csv",header=F)
interSampCov0 = apply(interCov0, 2, effectiveSize)

boxplot(whiteSampCov0,centerSampCov0,interSampCov0,names=c("white","center","inter"), main="Covariance 0 (scale = 0.1)")



### COV 0p1 

whiteCov0p1 = read.csv("whitecov0p10p1.csv",header=F)
whiteSampCov0p1 = apply(whiteCov0p1, 2, effectiveSize)

centerCov0p1 = read.csv("centercov0p10p1.csv",header=F)
centerSampCov0p1 = apply(centerCov0p1, 2, effectiveSize)

interCov0p1 = read.csv("intercov0p10p1.csv",header=F)
interSampCov0p1 = apply(interCov0p1, 2, effectiveSize)

boxplot(whiteSampCov0p1,centerSampCov0p1,interSampCov0p1,names=c("white","center","inter"), main="Covariance 0.1 (scale = 0.1)")


### prange1 

whiteprange1 = read.csv("whiteprange10p1.csv",header=F)
whiteSampprange1 = apply(whiteprange1, 2, effectiveSize)

centerprange1 = read.csv("centerprange10p1.csv",header=F)
centerSampprange1 = apply(centerprange1, 2, effectiveSize)

interprange1 = read.csv("interprange10p1.csv",header=F)
interSampprange1 = apply(interprange1, 2, effectiveSize)

boxplot(whiteSampprange1,centerSampprange1,interSampprange1,names=c("white","center","inter"), main="Practical Range 1 (scale = 0.1)")



### prange2

whiteprange2 = read.csv("whiteprange20p1.csv",header=F)
whiteSampprange2 = apply(whiteprange2, 2, effectiveSize)

centerprange2 = read.csv("centerprange20p1.csv",header=F)
centerSampprange2 = apply(centerprange2, 2, effectiveSize)

interprange2 = read.csv("interprange20p1.csv",header=F)
interSampprange2= apply(interprange2, 2, effectiveSize)

boxplot(whiteSampprange2,centerSampprange2,interSampprange2,names=c("white","center","inter"), main="Practical Range 2 (scale = 0.1)")


### scale 1 ################

### COV 0 

whiteCov0 = read.csv("whitecov01.csv",header=F)
whiteSampCov0 = apply(whiteCov0, 2, effectiveSize)

centerCov0 = read.csv("centercov01.csv",header=F)
centerSampCov0 = apply(centerCov0, 2, effectiveSize)

interCov0 = read.csv("intercov01.csv",header=F)
interSampCov0 = apply(interCov0, 2, effectiveSize)

boxplot(whiteSampCov0,centerSampCov0,interSampCov0,names=c("white","center","inter"), main="Covariance 0 (scale = 1)")



### COV 0p1 

whiteCov0p1 = read.csv("whitecov0p11.csv",header=F)
whiteSampCov0p1 = apply(whiteCov0p1, 2, effectiveSize)

centerCov0p1 = read.csv("centercov0p11.csv",header=F)
centerSampCov0p1 = apply(centerCov0p1, 2, effectiveSize)

interCov0p1 = read.csv("intercov0p11.csv",header=F)
interSampCov0p1 = apply(interCov0p1, 2, effectiveSize)

boxplot(whiteSampCov0p1,centerSampCov0p1,interSampCov0p1,names=c("white","center","inter"), main="Covariance 0.1 (scale = 1)")


### prange1 

whiteprange1 = read.csv("whiteprange11.csv",header=F)
whiteSampprange1 = apply(whiteprange1, 2, effectiveSize)

centerprange1 = read.csv("centerprange11.csv",header=F)
centerSampprange1 = apply(centerprange1, 2, effectiveSize)

interprange1 = read.csv("interprange11.csv",header=F)
interSampprange1 = apply(interprange1, 2, effectiveSize)

boxplot(whiteSampprange1,centerSampprange1,interSampprange1,names=c("white","center","inter"), main="Practical Range 1 (scale = 1)")



### prange2

whiteprange2 = read.csv("whiteprange21.csv",header=F)
whiteSampprange2 = apply(whiteprange2, 2, effectiveSize)

centerprange2 = read.csv("centerprange21.csv",header=F)
centerSampprange2 = apply(centerprange2, 2, effectiveSize)

interprange2 = read.csv("interprange21.csv",header=F)
interSampprange2= apply(interprange2, 2, effectiveSize)

boxplot(whiteSampprange2,centerSampprange2,interSampprange2,names=c("white","center","inter"), main="Practical Range 2 (scale = 1)")


### scale 10 ################

### COV 0 

whiteCov0 = read.csv("whitecov010.csv",header=F)
whiteSampCov0 = apply(whiteCov0, 2, effectiveSize)

centerCov0 = read.csv("centercov010.csv",header=F)
centerSampCov0 = apply(centerCov0, 2, effectiveSize)

interCov0 = read.csv("intercov010.csv",header=F)
interSampCov0 = apply(interCov0, 2, effectiveSize)

boxplot(whiteSampCov0,centerSampCov0,interSampCov0,names=c("white","center","inter"), main="Covariance 0 (scale = 10)")



### COV 0p1 

whiteCov0p1 = read.csv("whitecov0p110.csv",header=F)
whiteSampCov0p1 = apply(whiteCov0p1, 2, effectiveSize)

centerCov0p1 = read.csv("centercov0p110.csv",header=F)
centerSampCov0p1 = apply(centerCov0p1, 2, effectiveSize)

interCov0p1 = read.csv("intercov0p110.csv",header=F)
interSampCov0p1 = apply(interCov0p1, 2, effectiveSize)

boxplot(whiteSampCov0p1,centerSampCov0p1,interSampCov0p1,names=c("white","center","inter"), main="Covariance 0.1 (scale = 10)")


### prange1 

whiteprange1 = read.csv("whiteprange110.csv",header=F)
whiteSampprange1 = apply(whiteprange1, 2, effectiveSize)

centerprange1 = read.csv("centerprange110.csv",header=F)
centerSampprange1 = apply(centerprange1, 2, effectiveSize)

interprange1 = read.csv("interprange110.csv",header=F)
interSampprange1 = apply(interprange1, 2, effectiveSize)

boxplot(whiteSampprange1,centerSampprange1,interSampprange1,names=c("white","center","inter"), main="Practical Range 1 (scale = 10)")



### prange2

whiteprange2 = read.csv("whiteprange210.csv",header=F)
whiteSampprange2 = apply(whiteprange2, 2, effectiveSize)

centerprange2 = read.csv("centerprange210.csv",header=F)
centerSampprange2 = apply(centerprange2, 2, effectiveSize)

interprange2 = read.csv("interprange210.csv",header=F)
interSampprange2= apply(interprange2, 2, effectiveSize)

boxplot(whiteSampprange2,centerSampprange2,interSampprange2,names=c("white","center","inter"), main="Practical Range 2 (scale = 10)")




### Doing neat ggplot for paper


whiteCov00p1 = read.csv("whitecov00p1.csv",header=F)
whiteSampCov00p1 = apply(whiteCov00p1, 2, effectiveSize)

whiteCov0p10p1 = read.csv("whitecov0p10p1.csv",header=F)
whiteSampCov0p10p1 = apply(whiteCov0p10p1, 2, effectiveSize)


whiteCov01 = read.csv("whitecov01.csv",header=F)
whiteSampCov01 = apply(whiteCov01, 2, effectiveSize)

whiteCov0p11 = read.csv("whitecov0p11.csv",header=F)
whiteSampCov0p11 = apply(whiteCov0p11, 2, effectiveSize)


whiteCov010 = read.csv("whitecov010.csv",header=F)
whiteSampCov010 = apply(whiteCov010, 2, effectiveSize)

whiteCov0p110 = read.csv("whitecov0p110.csv",header=F)
whiteSampCov0p110 = apply(whiteCov0p110, 2, effectiveSize)


centerCov00p1 = read.csv("centercov00p1.csv",header=F)
centerSampCov00p1 = apply(centerCov00p1, 2, effectiveSize)

centerCov0p10p1 = read.csv("centercov0p10p1.csv",header=F)
centerSampCov0p10p1 = apply(centerCov0p10p1, 2, effectiveSize)


centerCov01 = read.csv("centercov01.csv",header=F)
centerSampCov01 = apply(centerCov01, 2, effectiveSize)

centerCov0p11 = read.csv("centercov0p11.csv",header=F)
centerSampCov0p11 = apply(centerCov0p11, 2, effectiveSize)


centerCov010 = read.csv("centercov010.csv",header=F)
centerSampCov010 = apply(centerCov010, 2, effectiveSize)

centerCov0p110 = read.csv("centercov0p110.csv",header=F)
centerSampCov0p110 = apply(centerCov0p110, 2, effectiveSize)


interCov00p1 = read.csv("intercov00p1.csv",header=F)
interSampCov00p1 = apply(interCov00p1, 2, effectiveSize)

interCov0p10p1 = read.csv("intercov0p10p1.csv",header=F)
interSampCov0p10p1 = apply(interCov0p10p1, 2, effectiveSize)


interCov01 = read.csv("intercov01.csv",header=F)
interSampCov01 = apply(interCov01, 2, effectiveSize)

interCov0p11 = read.csv("intercov0p11.csv",header=F)
interSampCov0p11 = apply(interCov0p11, 2, effectiveSize)


interCov010 = read.csv("intercov010.csv",header=F)
interSampCov010 = apply(interCov010, 2, effectiveSize)

interCov0p110 = read.csv("intercov0p110.csv",header=F)
interSampCov0p110 = apply(interCov0p110, 2, effectiveSize)



### format data



whiteDatCov00p1 = data.frame(res = whiteSampCov00p1, method = "Whitened", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 0.1")
whiteDatCov0p10p1 = data.frame(res = whiteSampCov0p10p1, method = "Whitened", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 0.1")

whiteDatCov01 = data.frame(res = whiteSampCov01, method = "Whitened", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 1")
whiteDatCov0p11 = data.frame(res = whiteSampCov0p11, method = "Whitened", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 1")

whiteDatCov010 = data.frame(res = whiteSampCov010, method = "Whitened", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 10")
whiteDatCov0p110 = data.frame(res = whiteSampCov0p110, method = "Whitened", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 10")


centerDatCov00p1 = data.frame(res = centerSampCov00p1, method = "Centered", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 0.1")
centerDatCov0p10p1 = data.frame(res = centerSampCov0p10p1, method = "Centered", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 0.1")

centerDatCov01 = data.frame(res = centerSampCov01, method = "Centered", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 1")
centerDatCov0p11 = data.frame(res = centerSampCov0p11, method = "Centered", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 1")

centerDatCov010 = data.frame(res = centerSampCov010, method = "Centered", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 10")
centerDatCov0p110 = data.frame(res = centerSampCov0p110, method = "Centered", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 10")


interDatCov00p1 = data.frame(res = interSampCov00p1, method = "Interweaved", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 0.1")
interDatCov0p10p1 = data.frame(res = interSampCov0p10p1, method = "Interweaved", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 0.1")

interDatCov01 = data.frame(res = interSampCov01, method = "Interweaved", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 1")
interDatCov0p11 = data.frame(res = interSampCov0p11, method = "Interweaved", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 1")

interDatCov010 = data.frame(res = interSampCov010, method = "Interweaved", parameter = "Cross-covariance (d=0)", scale="StN Ratio = 10")
interDatCov0p110 = data.frame(res = interSampCov0p110, method = "Interweaved", parameter = "Cross-covariance (d=0.1)", scale="StN Ratio = 10")

resData = rbind(whiteDatCov00p1,whiteDatCov0p10p1,whiteDatCov01,whiteDatCov0p11,whiteDatCov010,whiteDatCov0p110,
                centerDatCov00p1,centerDatCov0p10p1,centerDatCov01,centerDatCov0p11,centerDatCov010,centerDatCov0p110,
                interDatCov00p1,interDatCov0p10p1,interDatCov01,interDatCov0p11,interDatCov010,interDatCov0p110)

require(tidyverse)

resData = as.tibble(resData)

p <- ggplot(resData, aes(x=method, y=res, fill=method)) + 
  geom_boxplot() + ylab("Effective Sample Size") + xlab("") + guides(fill="none")  +
   theme_bw() + theme(axis.title.x=element_blank()) + facet_wrap(scale ~ parameter,scales="free_y", nrow=3)
ggsave(
  "effSamp.pdf",
  plot = p,
  device = "pdf", width = 6, height=8
)

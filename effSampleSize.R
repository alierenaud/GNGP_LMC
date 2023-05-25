


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


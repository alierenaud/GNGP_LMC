


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






### COV 0 

whiteCov0 = read.csv("whitecov0.csv",header=F)
whiteSampCov0 = apply(whiteCov0, 2, effectiveSize)

centerCov0 = read.csv("centercov0.csv",header=F)
centerSampCov0 = apply(centerCov0, 2, effectiveSize)

interCov0 = read.csv("intercov0.csv",header=F)
interSampCov0 = apply(interCov0, 2, effectiveSize)

boxplot(whiteSampCov0,centerSampCov0,interSampCov0,names=c("white","center","inter"), main="Covariance 0")



### COV 0p1 

whiteCov0p1 = read.csv("whitecov0p1.csv",header=F)
whiteSampCov0p1 = apply(whiteCov0p1, 2, effectiveSize)

centerCov0p1 = read.csv("centercov0p1.csv",header=F)
centerSampCov0p1 = apply(centerCov0p1, 2, effectiveSize)

interCov0p1 = read.csv("intercov0p1.csv",header=F)
interSampCov0p1 = apply(interCov0p1, 2, effectiveSize)

boxplot(whiteSampCov0p1,centerSampCov0p1,interSampCov0p1,names=c("white","center","inter"), main="Covariance 0.1")


### prange1 

whiteprange1 = read.csv("whiteprange1.csv",header=F)
whiteSampprange1 = apply(whiteprange1, 2, effectiveSize)

centerprange1 = read.csv("centerprange1.csv",header=F)
centerSampprange1 = apply(centerprange1, 2, effectiveSize)

interprange1 = read.csv("interprange1.csv",header=F)
interSampprange1 = apply(interprange1, 2, effectiveSize)

boxplot(whiteSampprange1,centerSampprange1,interSampprange1,names=c("white","center","inter"), main="Practical Range 1")



### prange2

whiteprange2 = read.csv("whiteprange2.csv",header=F)
whiteSampprange2 = apply(whiteprange2, 2, effectiveSize)

centerprange2 = read.csv("centerprange2.csv",header=F)
centerSampprange2 = apply(centerprange2, 2, effectiveSize)

interprange2 = read.csv("interprange2.csv",header=F)
interSampprange2= apply(interprange2, 2, effectiveSize)

boxplot(whiteSampprange2,centerSampprange2,interSampprange2,names=c("white","center","inter"), main="Practical Range 2")





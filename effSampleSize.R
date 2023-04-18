


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




centered = read.csv("centered.csv",header=F)

n = ncol(centered)
centerSamp = rep(0,n)

for (i in 1:n) {
  centerSamp[i] = effectiveSize(centered[,i])
}

boxplot(centerSamp)





white = read.csv("white.csv",header=F)

n = ncol(white)
whiteSamp = rep(0,n)

for (i in 1:n) {
  whiteSamp[i] = effectiveSize(white[,i])
}

boxplot(whiteSamp)




interweave = read.csv("interweave.csv",header=F)

n = ncol(interweave)
interSamp = rep(0,n)

for (i in 1:n) {
  interSamp[i] = effectiveSize(interweave[,i])
}

boxplot(interSamp)







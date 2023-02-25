


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




stand = read.csv("standard.csv",header=F)

effectiveSize(stand[,1])
effectiveSize(stand[,2])
effectiveSize(stand[,3])
effectiveSize(stand[,4])


inter = read.csv("interweave.csv",header=F)

effectiveSize(inter[,1])
effectiveSize(inter[,2])
effectiveSize(inter[,3])
effectiveSize(inter[,4])







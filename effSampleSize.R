


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

effectiveSize(centered[,1])
effectiveSize(centered[,2])
effectiveSize(centered[,3])
effectiveSize(centered[,4])


white = read.csv("white.csv",header=F)

effectiveSize(white[,1])
effectiveSize(white[,2])
effectiveSize(white[,3])
effectiveSize(white[,4])

interweave = read.csv("interweave.csv",header=F)

effectiveSize(interweave[,1])
effectiveSize(interweave[,2])
effectiveSize(interweave[,3])
effectiveSize(interweave[,4])







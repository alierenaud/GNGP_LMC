
require(tidyverse)

dMSE = read.csv("dMSE2.csv",header=F)


dMSE = t(dMSE)

colnames(dMSE) = c("Triangular", "Full",  "Diagonal")


dMSE = pivot_longer(as.data.frame(dMSE),c(1,2,3),names_to = "model")


p <- ggplot(dMSE, aes(x=model, y=value, fill=model)) + 
  geom_boxplot()  + ylab("MSE Difference") + xlab("") + guides(fill="none")  +
  theme_bw() + theme(axis.title.x=element_blank()) #+ facet_wrap(scale ~ parameter,scales="free_y", nrow=3)
ggsave(
  "MSEdiff2.pdf",
  plot = p,
  device = "pdf", width = 6, height=3
)

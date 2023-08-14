require(tidyverse)

dWnorms = read.csv("dWnorms.csv",header=F)


dWnorms = t(dWnorms)

colnames(dWnorms) = c("Triangular", "Full", "Block Diagonal", "Diagonal")


dWnorms = pivot_longer(as.data.frame(dWnorms),c(1,2,3,4),names_to = "model")


p <- ggplot(dWnorms, aes(x=model, y=value, fill=model)) + 
  geom_boxplot() + ylim(-1,0.5)  +ylab("Mean Distance Difference") + xlab("") + guides(fill="none")  +
  theme_bw() + theme(axis.title.x=element_blank()) #+ facet_wrap(scale ~ parameter,scales="free_y", nrow=3)
ggsave(
  "MDdiff.pdf",
  plot = p,
  device = "pdf", width = 6, height=3
)


require(tidyverse)

res_vec = read.csv("res_vec.csv", header = F)
res_mat = read.csv("res_mat.csv", header = F)

colnames(res_vec) = c("2","4","6","8")
colnames(res_mat) = c("2","4","6","8")

res_vec = cbind(res_vec,n = c("n = 100","n = 200","n = 300","n = 400"))
res_mat = cbind(res_mat,n = c("n = 100","n = 200","n = 300","n = 400"))


res_vec = res_vec %>% pivot_longer(cols = c("2","4","6","8"),names_to = "p", values_to = "time") %>%
                      mutate(Form="Vector")
res_mat = res_mat %>% pivot_longer(cols = c("2","4","6","8"),names_to = "p", values_to = "time") %>%
                      mutate(Form = "Matrix")


res_glob = rbind(res_vec,res_mat)

res_glob = res_glob %>% mutate(p = as.numeric(p))

p = ggplot(data=res_glob, aes(x=p, y=time, col=Form)) + facet_wrap( ~ n, nrow = 2, scales = "free")+
  geom_line()+
  geom_point() + ylab("Time (sec)")
p

ggsave(
  "timeCompa.pdf",
  plot = p,
  device = "pdf", width = 6, height=6
)
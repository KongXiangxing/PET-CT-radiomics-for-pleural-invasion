library(pROC)
library(ggplot2)

df_t <- read.csv(file = "./roc_t.csv",header = TRUE)

roc1t <- roc(df_t$Y_cli,df_t$cli, 
            smooth = F      
) 
roc2t <- roc(df_t$Y_ct,df_t$ct, 
            smooth = F       
)
roc3t <- roc(df_t$Y_pet,df_t$pet, 
            smooth = F       
)

roc.test(roc1t,roc2t,method = 'delong')
roc.test(roc1t,roc3t,method = 'delong')
roc.test(roc2t,roc3t,method = 'delong')



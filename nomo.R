pbc<-read.csv(file = "./nomo.csv",header = TRUE)
vali <- read.csv(file = "./nomo_vali.csv",header = TRUE)
head(pbc)
head(vali)
library(rms)
library(Hmisc)
library(grid)
library(lattice)
library(Formula)

library(ggplot2) 

dd<-datadist(pbc)
options(datadist="dd")
summary(pbc$label)
f1 <- lrm(label~ CYFRA21_1+NSE+SCC+Rad_score, data = pbc) 
nom <- nomogram(f1, fun= function(x)1/(1+exp(-x)),
                lp=F, funlabel="pleural invasion")
plot(nom)

f2 <- lrm(label~ CYFRA21_1+NSE+SCC++Rad_score, data = pbc, x=T, y=T)
f2
cal1 <- calibrate(f2, method = "boot",B=1000)


plot(cal1,
     xlim = c(0,1),
     xlab = "Predicted Probability",
     ylab = "Observed Probability",
     legend = FALSE,
     subtitles = FALSE)
abline(0,1,col = "black",lty = 2,lwd = 2)
lines(cal1[,c("predy","calibrated.orig")], type = "l",lwd = 2,col="red",pch =16)
lines(cal1[,c("predy","calibrated.corrected")], type = "l",lwd = 2,col="blue",pch =16)
legend(0.7,0.3,
       c("Ideal","Apparent","Bias-corrected"),
       lty = c(2,1,1),
       lwd = c(2,1,1),
       col = c("black","red","blue"),
       bty = "n",cex = 1.5)


pred.logit = predict(f2, newdata = test_name)
y_pred <- 1/(1+exp(-pred.logit))
val.prob(y_pred,as.numeric(test_name$pleural_invasion),m=20,cex=0.5)



library(rmda)
dca <- decision_curve(formula = label~CYFRA21_1+NSE+SCC++Rad_score, data = pbc,
                      family = binomial(link = "logit"),
                      confidence.intervals = F,bootstraps = 10,fitted.risk = F)
plot_decision_curve(dca,curve.names = "dca model",legend.position = "none")
legend("topright",legend = c("DCA","All","None"),col = c("red","grey","black"),
       lwd = c(3,1,2),cex = 1.5,bty = "n")




ddd<-datadist(vali)
options(datadist="ddd")
summary(vali$label)

f3 <- lrm(label~CYFRA21_1+NSE+SCC++Rad_score, data = vali, x=T, y=T)
f3
cal2 <- calibrate(f3, method = "boot",B=1000)
plot(cal2,
     xlim = c(0,1),
     xlab = "Predicted Probability",
     ylab = "Observed Probability",
     legend = FALSE,
     subtitles = FALSE)
abline(0,1,col = "black",lty = 2,lwd = 2)
lines(cal2[,c("predy","calibrated.orig")], type = "l",lwd = 2,col="red",pch =16)
lines(cal2[,c("predy","calibrated.corrected")], type = "l",lwd = 2,col="blue",pch =16)
legend(0.7,0.3,
       c("Ideal","Apparent","Bias-corrected"),
       lty = c(2,1,1),
       lwd = c(2,1,1),
       col = c("black","red","blue"),
       bty = "n",cex = 1.5)



library(rmda)
dca1 <- decision_curve(formula = label~CYFRA21_1+NSE+SCC++Rad_score, data = vali,
                      family = binomial(link = "logit"),
                      confidence.intervals = F,bootstraps = 10,fitted.risk = F)
plot_decision_curve(dca1,curve.names = "dca model",legend.position = "none")
legend("topright",legend = c("DCA","All","None"),col = c("red","grey","black"),
       lwd = c(3,1,2),cex = 1.5,bty = "n")


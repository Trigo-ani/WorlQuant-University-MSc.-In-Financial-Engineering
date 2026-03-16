################################
#
# Module Three - Lesson 2
#
################################

# Import libraries
library(moments)
library(plyr)
library(sn)
library(stats)

# Set the directory (please modify as required)
setwd('/WQU/Econometrics/module 3/data')

# Download the dataset
data_set <- read.csv(file = "M3. bond_and_stock_data.csv", header=TRUE, stringsAsFactors = FALSE)


#########################################
# Figure 5: Key Statistics for U.S. 10-Year Treasury Bond Yield Daily Return
#########################################

mean(data_set$X10Y_TBY)
sd(data_set$X10Y_TBY)
skewness(data_set$X10Y_TBY)
kurtosis(data_set$X10Y_TBY)



############################
# Figure 6: QQ Plot Normal Distribution vs U.S. 10-Year Treasury Bond Yield Daily Return
###########################

# Fit regular OLS
X10Y_nfit <- lm(X10Y_TBY ~ 1, data=data_set)

# QQ plot of residuals
plot(X10Y_nfit, which=2)



############################
# Figure 7: QQ Plot Skew-Normal Distribution vs U.S. Treasury 10-Year Bond Yield Daily Return
###########################

# Fit OLS with skew-Normal error terms
X10Y_snfit <- selm(X10Y_TBY ~ 1, data=data_set, family='SN')

# QQ plot of residuals
plot(X10Y_snfit, which=3, param.type='DP')



###############################
# Figure 8: Least Square Model Result
################################

# Fit regular OLS and print summary
ex_lm <- lm(X10Y_TBY ~ DWJ, data=data_set)
summary(ex_lm)



###############################
# Figure 9: Skew-Normal Regression Result
################################

# Fit OLS with skew normal error terms and print summary
ex_sn <- selm(X10Y_TBY ~ DWJ, data=data_set, family="SN")
summary(ex_sn, param.type='DP')



#########################################
# Figure 10: QQ Plots for Residuals from Normal Regression and for Residuals from Skew-Normal Regression
########################################

# QQ of residuals from regular regression
qqnorm(rstandard(ex_lm))

# QQ from skew-normal regression
plot(ex_sn, which=3)



##########################
# Figure 11: Simulated Skew-t Distributions with Different Degrees of Freedom
###########################

# skew t distribution with df=2000 almost infinity
set.seed(1234)
dpst1 <- cp2dp(c(25, 2, 0, 2000), family="ST")
st1 <- rst(3000, dp=dpst1)

# skew t distribution with df=6
set.seed(1234)
dpst2 <- cp2dp(c(25, 2, 0, 6), family="ST")
st2 <- rst(3000, dp=dpst2)

# skew t distribution with df=4
set.seed(1234)
dpst3 <- cp2dp(c(25, 2, 0, 4), family="ST")
st3 <- rst(3000, dp=dpst3)

# Plot 3 skew t distributions
plot(density(st1), xlab="X", ylim=c(0,0.3), main="Skew t Distribution")
lines(density(st2), col='red')
lines(density(st3), col='green')
legend("topright", inset=c(-0.005,0),  c("df=Infinity", "df=6", "df=4"),
       lty=rep(1, 3), bty="n", col=c("black","red","green"),
       pt.cex=1, cex=0.8, text.font=2, lwd=3)



##########################
# Figure 12: QQ Plot Skew-t Distribution vs. U.S. Treasury 10-Year Bond Yield Daily Return
###########################

# Fit OLS with skew t error terms
X10Y_stfit <- selm(X10Y_TBY ~ 1, data=data_set, family='ST')

# QQ plot of residuals
plot(X10Y_stfit, which=3, param.type='DP')



##########################
# Figure 13: Skew-t Distribution Regression Result
###########################

# Fit OLS with skew t error terms and print summary
ex_st <- selm(X10Y_TBY ~ DWJ, data=data_set, family="ST")
summary(ex_st, param.type='DP')



##########################
# Figure 14: QQ Plots for Residuals from Skew-Normal Regression and for Residuals from Skew-t Regression
###########################

plot(ex_st, which=3, param.type='DP')

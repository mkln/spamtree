# Spatial Multivariate Trees

This package implements the methods for Bayesian big data multivariate spatial regression appearing in Peruzzi and Dunson (2021). 
Estimation of all unknowns and predictions are carried out via Markov chain Monte Carlo within a single function. 
Refer to the documentation for the `spamtree` function and the examples. 

## Installing `spamtree`
Just do `install.packages("spamtree")` to install from CRAN.

## Documentation
A minimal documentation is available for the CRAN package and accessible from `R` via the usual `?spamtree::spamtree`.

## About `spamtree`
The methods implemented in the package are described in this article: 

### Spatial Multivariate Trees for Big Data Bayesian Regression
M Peruzzi and DB Dunson (2021), https://arxiv.org/abs/2012.00943

High resolution geospatial data are challenging because standard geostatistical models based on Gaussian processes are known to not scale to large data sizes. While progress has been made towards methods that can be computed more efficiently, considerably less attention has been devoted to big data methods that allow the description of complex relationships between several outcomes recorded at high resolutions by different sensors. Our Bayesian multivariate regression models based on spatial multivariate trees (SpamTrees) achieve scalability via conditional independence assumptions on latent random effects following a treed directed acyclic graph. Information-theoretic arguments and considerations on computational efficiency guide the construction of the tree and the related efficient sampling algorithms in imbalanced multivariate settings. In addition to simulated data examples, we illustrate SpamTrees using a large climate data set which combines satellite data with land-based station data.

## Example:

```r

library(abind)
library(magrittr)
library(tidyverse)
library(spamtree)

set.seed(2021)

SS <- 25 
n <- SS^2 # total n. locations, including missing ones

coords <- data.frame(Var1=runif(n), Var2=runif(n)) %>%
  as.matrix()

# generate data
sigmasq <- 2.3
phi <- 6
tausq <- .1
B <- c(-1,.5,1)

CC <- sigmasq * exp(-phi * as.matrix(dist(coords)))
LC <- t(chol(CC))
w <- LC %*% rnorm(n)
p <- length(B)
X <- rnorm(n * p) %>% matrix(ncol=p)
y_full <- X %*% B + w + tausq^.5 * rnorm(n)

set_missing <- rbinom(n, 1, 0.1)

simdata <- data.frame(coords,
                      y_full = y_full,
                      w_latent = w) %>%
  mutate(y_observed = ifelse(set_missing==1, NA, y_full))

y <- simdata$y_observed
ybar <- mean(y, na.rm=T)

# MCMC setup
mcmc_keep <- 1000
mcmc_burn <- 1000
mcmc_thin <- 2

# fit spamtree with defaults
spamtree_done <- spamtree(y - ybar, X, coords, 
                          mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin), 
                          num_threads = 10, verbose=TRUE)

# predictions
y_out <- spamtree_done$yhat_mcmc %>% 
  abind(along=3) %>% `[`(,1,) %>% add(ybar) %>% apply(1, mean)
w_out <- spamtree_done$w_mcmc %>% 
  abind(along=3) %>% `[`(,1,) %>% apply(1, mean)

outdf <- spamtree_done$coordsinfo %>% 
  cbind(data.frame(w_spamtree = w_out, 
                   y_spamtree = y_out)) %>%
  left_join(simdata)

# plot predictions
outdf %>% 
  ggplot(aes(Var1, Var2, color=y_spamtree)) +
  geom_point() +
  theme_minimal() +
  scale_color_viridis_c()

# rmspe
outdf %>% 
  filter(!complete.cases(.)) %>% 
  with((y_spamtree - y_full)^2) %>% 
  mean() %>% sqrt()

# plot latent process
outdf %>% 
  ggplot(aes(Var1, Var2, color=w_spamtree)) +
  geom_point() + 
  theme_minimal() +
  scale_color_viridis_c()

# estimation of regression coefficients
spamtree_done$beta_mcmc[,,1] %>% t() %>% 
  as.data.frame() %>% 
  gather(Coefficient, Sample) %>% 
  ggplot(aes(Sample)) + 
  geom_density() + 
  facet_wrap(~Coefficient, scales="free")


```

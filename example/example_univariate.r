rm(list=ls())
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
y <- X %*% B + w + tausq^.5 * rnorm(n)

set_missing <- rbinom(n, 1, 0.1)

simdata <- data.frame(coords,
                      y_full = y,
                      w_latent = w) %>%
  mutate(y_observed = ifelse(set_missing==1, NA, y_full))

# MCMC setup
mcmc_keep <- 1000
mcmc_burn <- 1000
mcmc_thin <- 2

ybar <- mean(y, na.rm=T)

# fit spamtree with defaults
spamtree_done <- spamtree(y - ybar, X, coords, 
                          mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin), 
                          num_threads = 10)

# predictions
y_out <- spamtree_done$yhat_mcmc %>% abind::abind(along=3) %>% `[`(,1,) %>% add(ybar) %>% apply(1, mean)
w_out <- spamtree_done$w_mcmc %>% abind::abind(along=3) %>% `[`(,1,) %>% apply(1, mean)

outdf <- spamtree_done$coordsinfo %>% 
  cbind(data.frame(w_spamtree = w_out, 
                   y_spamtree = y_out)) %>%
  left_join(simdata)

# plot predictions
outdf %>% 
  ggplot(aes(Var1, Var2, color=y_mvmr)) +
  geom_point() +
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
  scale_color_viridis_c()

# estimation of regression coefficients
spamtree_done$beta_mcmc[,,1] %>% t() %>% 
  as.data.frame() %>% 
  gather(Coefficient, Sample) %>% 
  ggplot(aes(Sample)) + 
  geom_density() + 
  facet_wrap(~Coefficient, scales="free")
